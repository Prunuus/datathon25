"""Evaluate a loaded DecisionNet policy for both starting corners to detect residual bias.

Runs games from both Player 1 and Player 2 perspectives (if --both_roles) and reports
separate win/draw/loss rates to highlight any positional advantage.

Usage:
    python evaluate_selfplay.py --weights weights2.pt --games 200 --both_roles
"""
import argparse
import os
import random
import torch
from case_closed_game import Game, Direction, GameResult
from model import load_model, build_features, DIRECTIONS


def select_move_with_model(state: dict, player_number: int, model, input_dim: int):
    # Compute features
    feats = build_features(state, player_number)
    if feats.numel() != input_dim:
        # fallback: random safe
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        return random.choice(dirs), False
    with torch.no_grad():
        dir_logits, boost_logit = model(feats.unsqueeze(0))
        probs = torch.softmax(dir_logits, dim=-1).squeeze(0).tolist()
        boost_p = torch.sigmoid(boost_logit).item()
    # pick highest-prob safe move
    board = state['board']
    H, W = len(board), len(board[0]) if board else (18, 20)
    my_trail = state['agent1_trail'] if player_number == 1 else state['agent2_trail']
    opp_trail = state['agent2_trail'] if player_number == 1 else state['agent1_trail']
    occ = set(tuple(p) for p in my_trail) | set(tuple(p) for p in opp_trail)
    head = my_trail[-1] if my_trail else (1, 2)

    # determine current direction to avoid invalid immediate reversals
    def current_dir(trail):
        if len(trail) < 2:
            return Direction.RIGHT
        (x1, y1), (x2, y2) = trail[-2], trail[-1]
        dx = x2 - x1
        dy = y2 - y1
        if dx > 1: dx = -1
        if dx < -1: dx = 1
        if dy > 1: dy = -1
        if dy < -1: dy = 1
        if (dx, dy) == (0, -1): return Direction.UP
        if (dx, dy) == (0, 1): return Direction.DOWN
        if (dx, dy) == (-1, 0): return Direction.LEFT
        return Direction.RIGHT

    cur_dir = current_dir(my_trail)
    opposite = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}

    def next_head(pos, d):
        dx, dy = d.value
        return ((pos[0]+dx) % W, (pos[1]+dy) % H)

    ranked = sorted(DIRECTIONS, key=lambda d: probs[DIRECTIONS.index(d)], reverse=True)
    # filter out the immediate opposite move unless all others blocked
    ranked_dirs = [getattr(Direction, d) for d in ranked]
    filtered = [d for d in ranked_dirs if d != opposite.get(cur_dir)] or ranked_dirs

    boosts_remaining = state.get('agent1_boosts', 3) if player_number == 1 else state.get('agent2_boosts', 3)

    def path_safe(start_head, direction, steps):
        h = start_head
        for _ in range(steps):
            h = next_head(h, direction)
            if h in occ:
                return False
        return True

    use_boost_allowed = boost_p > 0.55 and boosts_remaining > 0

    for d in filtered:
        nh = next_head(head, d)
        if nh in occ:
            continue
        # prefer boost only if entire two-step path remains safe
        if use_boost_allowed and path_safe(head, d, 2):
            return d, True
        return d, False
    # if all blocked, return top anyway
    return filtered[0], False


def strong_opp_move(state: dict, player_number: int):
    """Lightweight opponent: avoid immediate reversal and prefer safe random moves."""
    board = state['board']
    H, W = len(board), len(board[0]) if board else (18, 20)
    trail = state['agent2_trail'] if player_number == 2 else state['agent1_trail']
    other_trail = state['agent1_trail'] if player_number == 2 else state['agent2_trail']
    occ = set(tuple(p) for p in trail) | set(tuple(p) for p in other_trail)
    head = trail[-1] if trail else (1, 2)

    def next_head(pos, d):
        dx, dy = d.value
        return ((pos[0] + dx) % W, (pos[1] + dy) % H)

    def current_dir(tr):
        if len(tr) < 2:
            return Direction.RIGHT
        (x1, y1), (x2, y2) = tr[-2], tr[-1]
        dx = x2 - x1
        dy = y2 - y1
        if dx > 1: dx = -1
        if dx < -1: dx = 1
        if dy > 1: dy = -1
        if dy < -1: dy = 1
        if (dx, dy) == (0, -1): return Direction.UP
        if (dx, dy) == (0, 1): return Direction.DOWN
        if (dx, dy) == (-1, 0): return Direction.LEFT
        return Direction.RIGHT

    cur = current_dir(trail)
    opposite = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
    cand = [d for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT] if d != opposite[cur]]
    if not cand:
        cand = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    # prefer safe
    safe = [d for d in cand if next_head(head, d) not in occ]
    pool = safe if safe else cand
    return random.choice(pool)


def build_state(g: Game):
    return {
        'board': g.board.grid,
        'agent1_trail': g.agent1.get_trail_positions(),
        'agent2_trail': g.agent2.get_trail_positions(),
        'agent1_boosts': g.agent1.boosts_remaining,
        'agent2_boosts': g.agent2.boosts_remaining,
        'turn_count': g.turns,
    }


def play_one(model, input_dim: int, seed: int = 0, as_player: int = 1):
    random.seed(seed)
    g = Game()
    while True:
        s = build_state(g)
        if as_player == 1:
            d1, b1 = select_move_with_model(s, 1, model, input_dim)
            d2 = strong_opp_move(s, 2)
            res = g.step(d1, d2, b1, False)
        else:
            # Mirror evaluation: model controls Player 2
            d2, b2 = select_move_with_model(s, 2, model, input_dim)
            d1 = strong_opp_move(s, 1)
            res = g.step(d1, d2, False, b2)
        if res is not None:
            return res
        if g.turns >= 200:
            return GameResult.DRAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default=os.environ.get('MODEL_WEIGHTS', 'weights2.pt'))
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--both_roles', action='store_true', help='Evaluate model as Player 1 and Player 2 to gauge positional bias.')
    args = ap.parse_args()

    model, input_dim = load_model(args.weights)
    # Player 1 perspective
    p1_w = p1_d = p1_l = 0
    for i in range(args.games):
        res = play_one(model, input_dim, seed=i, as_player=1)
        if res == GameResult.AGENT1_WIN:
            p1_w += 1
        elif res == GameResult.AGENT2_WIN:
            p1_l += 1
        else:
            p1_d += 1

    print(f"P1 perspective over {args.games} games: W {p1_w} / D {p1_d} / L {p1_l}")

    if args.both_roles:
        p2_w = p2_d = p2_l = 0
        for i in range(args.games):
            res = play_one(model, input_dim, seed=10_000 + i, as_player=2)
            # When model is Player 2, a GameResult.AGENT2_WIN is a win for model.
            if res == GameResult.AGENT2_WIN:
                p2_w += 1
            elif res == GameResult.AGENT1_WIN:
                p2_l += 1
            else:
                p2_d += 1
        print(f"P2 perspective over {args.games} games: W {p2_w} / D {p2_d} / L {p2_l}")
        # Simple bias indicator: delta win rate
        wr_p1 = p1_w / max(1, args.games)
        wr_p2 = p2_w / max(1, args.games)
        print(f"Win rate delta (P1 - P2): {wr_p1 - wr_p2:.3f}")


if __name__ == '__main__':
    main()
