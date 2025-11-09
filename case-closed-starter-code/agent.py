import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import torch

from case_closed_game import Game, Direction, GameResult
from model import build_features, load_model, DIRECTIONS

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
_default_weights = os.path.join(os.path.dirname(__file__), "weights_bias.pt")
MODEL, MODEL_INPUT_DIM = load_model(os.environ.get("MODEL_WEIGHTS", _default_weights))
MODEL_ALPHA = float(os.environ.get("MODEL_ALPHA", "0.6"))  # model influence on scoring
MODEL_FORCE = os.environ.get("MODEL_FORCE", "0") == "1"   # prefer model top-1 among safe moves
AGENT_DEBUG = os.environ.get("AGENT_DEBUG", "0") == "1"

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentKevin"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining

    # ----------------- MODEL + HEURISTIC DECISION -------------------
    # If MODEL is available, compute feature vector and get logits.
    use_model = True
    model_dir_scores = None
    model_boost_score = None
    model_dir_probs = None
    try:
        feats = build_features(state, player_number=player_number)
        if feats.numel() != MODEL_INPUT_DIM:
            use_model = False
        else:
            with torch.no_grad():
                dir_logits, boost_logit = MODEL(feats.unsqueeze(0))
                model_dir_scores = dir_logits.squeeze(0)
                # Convert to probabilities for stable scaling
                model_dir_probs_t = torch.softmax(model_dir_scores, dim=-1)
                model_dir_probs = model_dir_probs_t.tolist()
                model_dir_scores = model_dir_scores.tolist()
                model_boost_score = boost_logit.item()
    except Exception as e:
        use_model = False
        if AGENT_DEBUG:
            print(f"[AGENT] Model inference error: {e}")

    # Helpers
    def board_dims(board):
        if not board or not isinstance(board, list):
            return 18, 20  # defaults from case_closed_game
        h = len(board)
        w = len(board[0]) if h > 0 else 20
        return h, w

    def norm(x, y, W, H):
        return (x % W, y % H)

    def dir_to_vec(d):
        if d == "UP":
            return (0, -1)
        if d == "DOWN":
            return (0, 1)
        if d == "LEFT":
            return (-1, 0)
        if d == "RIGHT":
            return (1, 0)
        return (1, 0)

    def vec_to_dir(dx, dy):
        if (dx, dy) == (0, -1):
            return "UP"
        if (dx, dy) == (0, 1):
            return "DOWN"
        if (dx, dy) == (-1, 0):
            return "LEFT"
        return "RIGHT"

    def current_direction_from_trail(trail, W, H):
        if len(trail) < 2:
            return "RIGHT"
        x2, y2 = trail[-1]
        x1, y1 = trail[-2]
        dx = (x2 - x1)
        dy = (y2 - y1)
        # normalize for wrap (minimal step)
        if dx > 1:
            dx = -1
        elif dx < -1:
            dx = 1
        if dy > 1:
            dy = -1
        elif dy < -1:
            dy = 1
        return vec_to_dir(dx, dy)

    def next_head(head, dstr, W, H):
        dx, dy = dir_to_vec(dstr)
        return norm(head[0] + dx, head[1] + dy, W, H)

    def trail_set(trail_list):
        return set(tuple(p) for p in trail_list)

    def count_safe_moves(head, obstacles, W, H, forbid_opposite=None):
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        if forbid_opposite:
            opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            if opp.get(forbid_opposite) in moves:
                moves.remove(opp[forbid_opposite])
        c = 0
        for m in moves:
            nh = next_head(head, m, W, H)
            if nh not in obstacles:
                c += 1
        return c

    def flood_fill_area(start, obstacles, W, H, max_cap=None):
        if start in obstacles:
            return 0
        seen = set([start])
        dq = deque([start])
        total = 0
        while dq:
            x, y = dq.popleft()
            total += 1
            if max_cap and total >= max_cap:
                return total
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = norm(x + dx, y + dy, W, H)
                if (nx, ny) not in obstacles and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    dq.append((nx, ny))
        return total

    def voronoi_advantage(my_head, opp_head, obstacles, W, H, sample_cap=None):
        #approximate cells closer to me vs opponent
        from_me = {my_head: 0}
        from_opp = {opp_head: 0}
        q_me = deque([my_head])
        q_opp = deque([opp_head])
        owned_me = 0
        owned_opp = 0
        visited = set()
        cap_count = 0
        while q_me or q_opp:
            if q_me:
                cx, cy = q_me.popleft()
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = norm(cx + dx, cy + dy, W, H)
                    if (nx, ny) in obstacles:
                        continue
                    if (nx, ny) not in from_me:
                        from_me[(nx, ny)] = from_me[(cx, cy)] + 1
                        q_me.append((nx, ny))
            if q_opp:
                cx, cy = q_opp.popleft()
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = norm(cx + dx, cy + dy, W, H)
                    if (nx, ny) in obstacles:
                        continue
                    if (nx, ny) not in from_opp:
                        from_opp[(nx, ny)] = from_opp[(cx, cy)] + 1
                        q_opp.append((nx, ny))
            cap_count += 1
            if sample_cap and cap_count > sample_cap:
                break
        # compare dist
        all_cells = set(from_me.keys()) | set(from_opp.keys())
        for c in all_cells:
            dm = from_me.get(c, 10**9)
            do = from_opp.get(c, 10**9)
            if dm < do:
                owned_me += 1
            elif do < dm:
                owned_opp += 1
        return owned_me - owned_opp

    def apply_move_once(head, dstr, obstacles, my_trail_set, opp_trail_set, W, H):
        nh = next_head(head, dstr, W, H)
        #collison logic
        if nh in my_trail_set or nh in opp_trail_set or nh in obstacles:
            return nh, False, True  # new head, alive, died
        return nh, True, False

    def evaluate_after(my_head, opp_head, obstacles, W, H, my_dir_prev, opp_dir_prev):
        #signals
        my_safe_next = count_safe_moves(my_head, obstacles, W, H, forbid_opposite=my_dir_prev)
        opp_safe_next = count_safe_moves(opp_head, obstacles, W, H, forbid_opposite=opp_dir_prev)

        # areas
        my_area = flood_fill_area(my_head, obstacles, W, H)
        opp_area = flood_fill_area(opp_head, obstacles, W, H)
        vor_adv = voronoi_advantage(my_head, opp_head, obstacles, W, H, sample_cap=200)

        # distance heuristic
        dx = min((my_head[0] - opp_head[0]) % W, (opp_head[0] - my_head[0]) % W)
        dy = min((my_head[1] - opp_head[1]) % H, (opp_head[1] - my_head[1]) % H)
        dist = dx + dy

        score = 0.0
        # match training-time coefficients for better alignment
        score += 3.0 * my_safe_next
        score -= 2.2 * opp_safe_next
        score += 0.4 * (my_area - opp_area)
        score += 0.9 * vor_adv
        score += 0.15 * dist
        return score

    # extract the board state
    board = state.get("board")
    H, W = board_dims(board)
    a1 = state.get("agent1_trail", [])
    a2 = state.get("agent2_trail", [])
    my_trail = a1 if player_number == 1 else a2
    opp_trail = a2 if player_number == 1 else a1
    my_head = tuple(my_trail[-1]) if my_trail else (1, 2)
    opp_head = tuple(opp_trail[-1]) if opp_trail else (W - 2, H - 3)

    # build obsticle for later sigmaa idk
    obstacles = set()
    if isinstance(board, list):
        for y in range(len(board)):
            row = board[y]
            for x in range(len(row)):
                if row[x] != 0:
                    obstacles.add((x, y))
    # add trails into obstacles
    for p in my_trail:
        obstacles.add(tuple(p))
    for p in opp_trail:
        obstacles.add(tuple(p))

    my_trail_set = trail_set(my_trail)
    opp_trail_set = trail_set(opp_trail)

    # prevent reversing
    my_prev_dir = current_direction_from_trail(my_trail, W, H)
    opp_prev_dir = current_direction_from_trail(opp_trail, W, H)
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    candidate_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
    if opposite.get(my_prev_dir) in candidate_dirs:
        candidate_dirs.remove(opposite[my_prev_dir])

    # trapped scenario, panic
    if not candidate_dirs:
        candidate_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]

    # try to predict oppent activites
    opp_cands = ["UP", "DOWN", "LEFT", "RIGHT"]
    if opposite.get(opp_prev_dir) in opp_cands:
        opp_cands.remove(opposite[opp_prev_dir])
    if not opp_cands:
        opp_cands = ["UP", "DOWN", "LEFT", "RIGHT"]

    if use_model and MODEL_FORCE and model_dir_probs is not None:
        ranked = sorted(DIRECTIONS, key=lambda d: model_dir_probs[DIRECTIONS.index(d)], reverse=True)
        H, W = board_dims(state.get("board"))
        my_head_tmp = tuple(my_trail[-1]) if my_trail else (1, 2)
        obstacles_tmp = set(tuple(p) for p in my_trail) | set(tuple(p) for p in opp_trail)
        for dmy in ranked:
            nhx, nhy = next_head(my_head_tmp, dmy, W, H)
            if (nhx, nhy) not in obstacles_tmp:
                use_boost = bool(model_boost_score is not None and model_boost_score > 0.5 and boosts_remaining > 0)
                move = dmy + (":BOOST" if use_boost else "")
                if AGENT_DEBUG:
                    print(f"[AGENT] MODEL_FORCE picked {move}; probs={model_dir_probs}; boost={model_boost_score:.3f}")
                return jsonify({"move": move}), 200

    # If model present, prefer exploring top-K model directions among safe candidates to reduce branching
    if use_model and model_dir_probs is not None:
        # keep relative order of best 3 model directions among candidates
        ranked = sorted(candidate_dirs, key=lambda d: model_dir_probs[DIRECTIONS.index(d)], reverse=True)
        topk = max(2, min(3, len(ranked)))
        candidate_dirs = ranked[:topk] + [d for d in candidate_dirs if d not in ranked[:topk]]

    # with/without boost scenario (blended scoring)
    best_move = "RIGHT"
    best_use_boost = False
    best_score = -10**9

    turn_count = state.get("turn_count", 0)

    for use_boost in [False, True]:
        if use_boost and boosts_remaining <= 0:
            continue
        # boost if needed
        for dmy in candidate_dirs:
            alive = True
            new_head = my_head
            new_obstacles = set(obstacles)
            steps = 2 if use_boost else 1
            for _ in range(steps):
                new_head, ok, died = apply_move_once(new_head, dmy, new_obstacles, my_trail_set, opp_trail_set, W, H)
                if died:
                    alive = False
                    break
                # trail growing
                new_obstacles.add(new_head)
            if not alive:
                worst_reply = -10**8
                if worst_reply > best_score:
                    best_score = worst_reply
                    best_move = dmy
                    best_use_boost = use_boost
                continue

            my_score_vs_best_opp = 10**9
            for dop in opp_cands:
                opp_alive = True
                opp_new_head = opp_head
                opp_steps = 1
                opp_obstacles = set(new_obstacles)
                for _ in range(opp_steps):
                    opp_new_head, ok, died = apply_move_once(opp_new_head, dop, opp_obstacles, opp_trail_set, my_trail_set, W, H)
                    if died:
                        opp_alive = False
                        break
                    opp_obstacles.add(opp_new_head)

                # if both land on the same head cell simultaneously, game logic marks draw; penalize draws slightly
                if opp_alive and opp_new_head == new_head:
                    # approximate draw
                    sc = -5000
                else:
                    # evaluate positions
                    sc = evaluate_after(new_head, opp_new_head if opp_alive else opp_head, opp_obstacles, W, H, dmy, dop)
                    # reward if opponent died
                    if not opp_alive:
                        sc += 5000

                # pessimistic 
                if sc < my_score_vs_best_opp:
                    my_score_vs_best_opp = sc

            # prefer survival over others
            # blend model directional preference (soft bonus) if available.
            final_score = my_score_vs_best_opp
            if use_model and model_dir_probs is not None:
                idx = DIRECTIONS.index(dmy)
                final_score += MODEL_ALPHA * model_dir_probs[idx]
            if final_score > best_score:
                best_score = final_score
                best_move = dmy
                best_use_boost = use_boost

    # boost usage gating: don't waste boosts early; use if it materially helps or we're in danger
    # if using boost doesn't improve score by margin, drop it unless we have few safe moves
    if best_use_boost:
        no_boost_score = -10**9
        for dmy in candidate_dirs:
            alive = True
            new_head = my_head
            new_obstacles = set(obstacles)
            new_head, ok, died = apply_move_once(new_head, dmy, new_obstacles, my_trail_set, opp_trail_set, W, H)
            if died:
                continue
            new_obstacles.add(new_head)
            my_score_vs_best_opp = 10**9
            for dop in opp_cands:
                opp_alive = True
                opp_new_head = opp_head
                opp_obstacles = set(new_obstacles)
                opp_new_head, ok, died = apply_move_once(opp_new_head, dop, opp_obstacles, opp_trail_set, my_trail_set, W, H)
                if died:
                    sc = 5000
                else:
                    opp_obstacles.add(opp_new_head)
                    sc = evaluate_after(new_head, opp_new_head, opp_obstacles, W, H, dmy, dop)
                if sc < my_score_vs_best_opp:
                    my_score_vs_best_opp = sc
            if my_score_vs_best_opp > no_boost_score:
                no_boost_score = my_score_vs_best_opp
        margin = best_score - no_boost_score
        immediate_safety = count_safe_moves(my_head, obstacles, W, H, forbid_opposite=my_prev_dir)
        # model boost preference: require both margin or strong boost logit
        boost_pref = (model_boost_score if (use_model and model_boost_score is not None) else -1.0)
        if (margin < 5.0 and boost_pref < 0.5) and immediate_safety >= 2 and turn_count < 50:
            best_use_boost = False

    move = best_move + (":BOOST" if best_use_boost else "")
    if AGENT_DEBUG and use_model:
        print(f"[AGENT] move={move} score={best_score:.2f} model_probs={model_dir_probs} boost_logit={model_boost_score}")
    # ----------------- END CODE HERE --------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=False)
