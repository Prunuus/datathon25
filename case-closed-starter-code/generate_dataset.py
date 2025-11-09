"""Self-play dataset generator for Case Closed (CPU-only).

Generates (features, label) pairs by running a stronger heuristic (minimax one-step
with flood-fill/Voronoi scoring) against a diverse opponent. Labels are the selected
direction (0..3) and a boost flag. We also augment states by rotations/reflections
to improve generalization.

Usage (example):
        python generate_dataset.py --episodes 400 --output dataset2.pt --opp strong

Output torch file contains:
{
    'X': FloatTensor [N, feature_dim],
    'y_dir': LongTensor [N],
    'y_boost': FloatTensor [N],  # 0/1
    'feature_dim': int
}
"""
import argparse
import torch
from collections import deque
from case_closed_game import Game, Direction
from model import build_features, DIRECTIONS, DIR_TO_IDX
import random

def _norm(x, y, W, H):
    return (x % W, y % H)

def _dir_to_vec(d):
    return d.value

def _vec_to_dir(dx, dy):
    if (dx, dy) == (0, -1):
        return Direction.UP
    if (dx, dy) == (0, 1):
        return Direction.DOWN
    if (dx, dy) == (-1, 0):
        return Direction.LEFT
    return Direction.RIGHT

def _next_head(head, d, W, H):
    dx, dy = _dir_to_vec(d)
    return _norm(head[0] + dx, head[1] + dy, W, H)

def _current_dir_from_trail(trail, W, H):
    if len(trail) < 2:
        return Direction.RIGHT
    (x1, y1), (x2, y2) = trail[-2], trail[-1]
    dx, dy = x2 - x1, y2 - y1
    if dx > 1: dx = -1
    if dx < -1: dx = 1
    if dy > 1: dy = -1
    if dy < -1: dy = 1
    return _vec_to_dir(dx, dy)

def _trail_set(trail):
    return set(tuple(p) for p in trail)

def _count_safe_moves(head, obstacles, W, H, forbid_opposite=None):
    moves = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if forbid_opposite is not None:
        opp = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
               Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
        bad = opp.get(forbid_opposite)
        if bad in moves:
            moves.remove(bad)
    c = 0
    for m in moves:
        nh = _next_head(head, m, W, H)
        if nh not in obstacles:
            c += 1
    return c

def _flood_fill_area(start, obstacles, W, H, max_cap=None):
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
            nx, ny = _norm(x + dx, y + dy, W, H)
            if (nx, ny) not in obstacles and (nx, ny) not in seen:
                seen.add((nx, ny))
                dq.append((nx, ny))
    return total

def _voronoi_advantage(my_head, opp_head, obstacles, W, H, sample_cap=None):
    from_me = {my_head: 0}
    from_opp = {opp_head: 0}
    q_me = deque([my_head])
    q_opp = deque([opp_head])
    owned_me = 0
    owned_opp = 0
    steps = 0
    while q_me or q_opp:
        if q_me:
            cx, cy = q_me.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = _norm(cx + dx, cy + dy, W, H)
                if (nx, ny) in obstacles or (nx, ny) in from_me:
                    continue
                from_me[(nx, ny)] = from_me[(cx, cy)] + 1
                q_me.append((nx, ny))
        if q_opp:
            cx, cy = q_opp.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = _norm(cx + dx, cy + dy, W, H)
                if (nx, ny) in obstacles or (nx, ny) in from_opp:
                    continue
                from_opp[(nx, ny)] = from_opp[(cx, cy)] + 1
                q_opp.append((nx, ny))
        steps += 1
        if sample_cap and steps > sample_cap:
            break
    all_cells = set(from_me.keys()) | set(from_opp.keys())
    for c in all_cells:
        dm = from_me.get(c, 10**9)
        do = from_opp.get(c, 10**9)
        if dm < do:
            owned_me += 1
        elif do < dm:
            owned_opp += 1
    return owned_me - owned_opp

def _apply_move_once(head, d, obstacles, my_trail_set, opp_trail_set, W, H):
    nh = _next_head(head, d, W, H)
    if nh in my_trail_set or nh in opp_trail_set or nh in obstacles:
        return nh, False, True
    return nh, True, False

def strong_heuristic_move(state, player_number):
    board = state.get('board')
    H, W = len(board), len(board[0]) if board else (18, 20)
    a1 = state.get('agent1_trail', [])
    a2 = state.get('agent2_trail', [])
    my_trail = a1 if player_number == 1 else a2
    opp_trail = a2 if player_number == 1 else a1
    my_head = tuple(my_trail[-1]) if my_trail else (1, 2)
    opp_head = tuple(opp_trail[-1]) if opp_trail else (W-2, H-3)

    obstacles = set()
    if isinstance(board, list):
        for y in range(len(board)):
            row = board[y]
            for x in range(len(row)):
                if row[x] != 0:
                    obstacles.add((x, y))
    for p in my_trail:
        obstacles.add(tuple(p))
    for p in opp_trail:
        obstacles.add(tuple(p))

    my_trail_set = _trail_set(my_trail)
    opp_trail_set = _trail_set(opp_trail)
    my_prev_dir = _current_dir_from_trail(my_trail, W, H)
    opp_prev_dir = _current_dir_from_trail(opp_trail, W, H)
    opposite = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}

    cand = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    bad = opposite.get(my_prev_dir)
    cand = [c for c in cand if c != bad] or [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    opp_cands = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    opp_cands = [c for c in opp_cands if c != opposite.get(opp_prev_dir)] or opp_cands

    def evaluate_after(my_h, opp_h, obst, my_dir_prev, opp_dir_prev):
        my_safe = _count_safe_moves(my_h, obst, W, H, forbid_opposite=my_dir_prev)
        opp_safe = _count_safe_moves(opp_h, obst, W, H, forbid_opposite=opp_dir_prev)
        my_area = _flood_fill_area(my_h, obst, W, H)
        opp_area = _flood_fill_area(opp_h, obst, W, H)
        vor = _voronoi_advantage(my_h, opp_h, obst, W, H, sample_cap=200)
        # wrapped manhattan
        dx = min((my_h[0]-opp_h[0]) % W, (opp_h[0]-my_h[0]) % W)
        dy = min((my_h[1]-opp_h[1]) % H, (opp_h[1]-my_h[1]) % H)
        dist = dx + dy
        score = 3.0*my_safe - 2.2*opp_safe + 0.4*(my_area-opp_area) + 0.9*vor + 0.15*dist
        return score

    best_dir = Direction.RIGHT
    best_use_boost = False
    best_score = -1e9

    boosts_remaining = state.get('agent1_boosts', 3) if player_number == 1 else state.get('agent2_boosts', 3)
    turn_count = state.get('turn_count', 0)

    for use_boost in [False, True]:
        if use_boost and boosts_remaining <= 0:
            continue
        for dmy in cand:
            alive = True
            new_head = my_head
            new_obst = set(obstacles)
            steps = 2 if use_boost else 1
            for _ in range(steps):
                new_head, ok, died = _apply_move_once(new_head, dmy, new_obst, my_trail_set, opp_trail_set, W, H)
                if died:
                    alive = False
                    break
                new_obst.add(new_head)
            if not alive:
                sc = -5e7
                if sc > best_score:
                    best_score = sc
                    best_dir = dmy
                    best_use_boost = use_boost
                continue
            worst_reply = 1e9
            for dop in opp_cands:
                opp_alive = True
                opp_new_head = opp_head
                opp_obst = set(new_obst)
                opp_new_head, ok, died = _apply_move_once(opp_new_head, dop, opp_obst, opp_trail_set, my_trail_set, W, H)
                if died:
                    sc = 5e3
                else:
                    opp_obst.add(opp_new_head)
                    sc = evaluate_after(new_head, opp_new_head, opp_obst, dmy, dop)
                if sc < worst_reply:
                    worst_reply = sc
            final = worst_reply
            if final > best_score:
                best_score = final
                best_dir = dmy
                best_use_boost = use_boost

    # boost gating similar to agent
    if best_use_boost:
        # only keep if margin is meaningful or in danger
        no_boost_score = -1e9
        for dmy in cand:
            new_head, ok, died = _apply_move_once(my_head, dmy, set(obstacles), my_trail_set, opp_trail_set, W, H)
            if died:
                continue
            tmp_ob = set(obstacles)
            tmp_ob.add(new_head)
            worst_reply = 1e9
            for dop in opp_cands:
                opp_new_head, ok, died = _apply_move_once(opp_head, dop, set(tmp_ob), opp_trail_set, my_trail_set, W, H)
                sc = 5e3 if died else evaluate_after(new_head, opp_new_head, set(tmp_ob)|{opp_new_head}, dmy, dop)
                if sc < worst_reply:
                    worst_reply = sc
            no_boost_score = max(no_boost_score, worst_reply)
        margin = best_score - no_boost_score
        immediate_safety = _count_safe_moves(my_head, obstacles, W, H, forbid_opposite=my_prev_dir)
        if margin < 5.0 and immediate_safety >= 2 and turn_count < 50:
            best_use_boost = False

    return best_dir, best_use_boost

def build_state_from_game(g: Game) -> dict:
    return {
        'board': g.board.grid,
        'agent1_trail': g.agent1.get_trail_positions(),
        'agent2_trail': g.agent2.get_trail_positions(),
        'agent1_boosts': g.agent1.boosts_remaining,
        'agent2_boosts': g.agent2.boosts_remaining,
        'turn_count': g.turns,
    }

def augment_state(state: dict, kind: str) -> dict:
    """Apply symmetry: rot90, rot180, rot270, flipx, flipy. 'none' returns original.
    We operate on board and trail coordinates under wrap-around geometry.
    """
    H = len(state['board'])
    W = len(state['board'][0])

    def rot90(x, y):
        return (y, (W-1)-x)
    def rot180(x, y):
        return ((W-1)-x, (H-1)-y)
    def rot270(x, y):
        return ((H-1)-y, x)
    def flipx(x, y):
        return ((W-1)-x, y)
    def flipy(x, y):
        return (x, (H-1)-y)

    def map_grid(mapper):
        g2 = [[0 for _ in range(W)] for _ in range(H)]
        for y in range(H):
            for x in range(W):
                nx, ny = mapper(x, y)
                g2[ny][nx] = state['board'][y][x]
        return g2

    def map_trail(tr, mapper):
        return [mapper(x, y) for (x, y) in tr]

    if kind == 'none':
        return state

    maps = {
        'rot90': rot90,
        'rot180': rot180,
        'rot270': rot270,
        'flipx': flipx,
        'flipy': flipy,
    }
    mapper = maps[kind]
    new_board = map_grid(mapper)
    s2 = dict(state)
    s2['board'] = new_board
    s2['agent1_trail'] = map_trail(state['agent1_trail'], mapper)
    s2['agent2_trail'] = map_trail(state['agent2_trail'], mapper)
    return s2

def remap_direction_for_augment(d: Direction, kind: str) -> Direction:
    """When applying symmetry to the board, the label direction should change too.
    - flipx: LEFT <-> RIGHT
    - flipy: UP <-> DOWN
    - rot180: reverse both axes (UP<->DOWN, LEFT<->RIGHT)
    """
    if kind == 'flipx':
        if d == Direction.LEFT:
            return Direction.RIGHT
        if d == Direction.RIGHT:
            return Direction.LEFT
        return d
    if kind == 'flipy':
        if d == Direction.UP:
            return Direction.DOWN
        if d == Direction.DOWN:
            return Direction.UP
        return d
    if kind == 'rot180':
        if d == Direction.UP:
            return Direction.DOWN
        if d == Direction.DOWN:
            return Direction.UP
        if d == Direction.LEFT:
            return Direction.RIGHT
        if d == Direction.RIGHT:
            return Direction.LEFT
        return d
    # 'none' or unhandled kinds
    return d


def run_episode(opp_kind: str = 'mixed', max_turns: int = 200):
    g = Game()
    samples = []
    for _ in range(max_turns):
        state = build_state_from_game(g)
        # Compute strong heuristic choices for both roles
        d1, b1 = strong_heuristic_move(state, player_number=1)
        d2, b2 = strong_heuristic_move(state, player_number=2)

        # Choose actual actions for each side depending on opponent style
        def random_not_opposite(cur):
            opposite = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
                        Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
            pool = [d for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT] if d != opposite[cur]]
            return random.choice(pool or [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])

        if opp_kind == 'random':
            a1 = random_not_opposite(g.agent1.direction)
            a2 = random_not_opposite(g.agent2.direction)
            b1_play = False
            b2_play = False
        elif opp_kind == 'strong':
            a1, b1_play = d1, bool(b1)
            a2, b2_play = d2, bool(b2)
        else:  # mixed
            a1 = (random_not_opposite(g.agent1.direction) if random.random() < 0.5 else d1)
            b1_play = (False if a1 != d1 else bool(b1))
            a2 = (random_not_opposite(g.agent2.direction) if random.random() < 0.5 else d2)
            b2_play = (False if a2 != d2 else bool(b2))

        g.step(a1, a2, b1_play, b2_play)

        # Collect samples according to role_mode
        # Restrict dataset to Player 1 (top-left start) only for training.
        samples.append((state, 1, d1, float(b1)))
        if not g.agent1.alive or not g.agent2.alive:
            break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--output', type=str, default='dataset2.pt')
    ap.add_argument('--opp', type=str, default='mixed', choices=['random','strong','mixed'])
    # Collect both player roles to remove starting-corner bias.
    ap.add_argument('--both_roles', action='store_true', help='Include samples from both Player 1 and Player 2 perspectives (recommended to reduce corner bias).')
    ap.add_argument('--augment', action='store_true', help='Apply symmetry augmentation (flips + 180 rotation) to reduce positional bias.')
    args = ap.parse_args()

    X = []
    y_dir = []
    y_boost = []

    # We can safely use flipx/flipy and rot180 (rot90/rot270 would swap dims 18x20 -> 20x18 and break fixed-size features).
    augments = ['none', 'flipx', 'flipy', 'rot180'] if args.augment else ['none']

    for ep in range(args.episodes):
        ep_samples = run_episode(opp_kind=args.opp)
        for state, role_player, d, b in ep_samples:
            # Always include Player 1 sample (role_player==1 by generation) and optionally Player 2 symmetric sample
            roles_to_add = [role_player]
            if args.both_roles:
                # Add the other player's perspective by recomputing heuristic direction for that role if not already collected
                if role_player == 1:
                    # Retrieve player2 heuristic move from state by calling strong_heuristic_move
                    d_other, b_other = strong_heuristic_move(state, player_number=2)
                    roles_to_add.append(2)
                    extra_dir_boost = {1: (d, b), 2: (d_other, float(b_other))}
                else:
                    d_other, b_other = strong_heuristic_move(state, player_number=1)
                    roles_to_add.append(1)
                    extra_dir_boost = {role_player: (d, b), 1: (d_other, float(b_other))}
            else:
                extra_dir_boost = {role_player: (d, b)}

            for r in roles_to_add:
                d_r, b_r = extra_dir_boost[r]
                # base feature
                X.append(build_features(state, player_number=r))
                y_dir.append(DIR_TO_IDX[d_r.name])
                y_boost.append(b_r)
                # augmentation (apply respecting role)
                for kind in augments:
                    if kind == 'none':
                        continue
                    s2 = augment_state(state, kind)
                    d_aug = remap_direction_for_augment(d_r, kind)
                    X.append(build_features(s2, player_number=r))
                    y_dir.append(DIR_TO_IDX[d_aug.name])
                    y_boost.append(b_r)
        if (ep+1) % 10 == 0:
            print(f"Collected episodes: {ep+1}")

    if not X:
        print("No samples collected.")
        return

    X_tensor = torch.stack(X)  # [N, feature_dim]
    y_dir_tensor = torch.tensor(y_dir, dtype=torch.long)
    y_boost_tensor = torch.tensor(y_boost, dtype=torch.float32)
    torch.save({'X': X_tensor, 'y_dir': y_dir_tensor, 'y_boost': y_boost_tensor, 'feature_dim': X_tensor.shape[1]}, args.output)
    print(f"Saved dataset with {X_tensor.shape[0]} samples to {args.output}")

if __name__ == '__main__':
    main()
