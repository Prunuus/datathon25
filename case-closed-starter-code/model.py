import os
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# (match case_closed_game defaults)
BOARD_H = 18
BOARD_W = 20

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}


def _one_hot(n: int, idx: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _wrap_dist(a: int, b: int, mod: int) -> int:
    return min((a - b) % mod, (b - a) % mod)


def build_features(state: dict, player_number: int) -> torch.Tensor:
    """Build a CPU tensor of features from judge state for the given player.

    Compact, CPU-friendly features combining global occupancy and local geometry, invariant to starting corner:
    - 2 channels occupancy from trails (mine, opp): 18*20*2 = 720
    - my/opp current dir one-hot (8)
    - boosts remaining (mine, opp) normalized /3 (2)
    - turn_count normalized (/500) (1)
    - 1-step and 2-step safe move fractions for me (2)
    - wrapped dx, dy distances normalized (/W, /H) (2)
    - flood-fill reachable area from my/opp heads normalized (/cells) (2)
    - Voronoi advantage approximation normalized (/cells) (1)
    - Ray distances to nearest obstacle in four cardinal directions from my/opp heads (/W or /H) (8)
    Total dims ~ 744.
    """
    board = state.get("board") or [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    a1 = state.get("agent1_trail", [])
    a2 = state.get("agent2_trail", [])
    my_trail = a1 if player_number == 1 else a2
    opp_trail = a2 if player_number == 1 else a1

    # head and direction from trail
    def current_dir(trail: List[Tuple[int, int]]) -> str:
        if len(trail) < 2:
            return "RIGHT"
        (x1, y1), (x2, y2) = trail[-2], trail[-1]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 1:
            dx = -1 if dx > 0 else 1
        if abs(dy) > 1:
            dy = -1 if dy > 0 else 1
        if (dx, dy) == (0, -1):
            return "UP"
        if (dx, dy) == (0, 1):
            return "DOWN"
        if (dx, dy) == (-1, 0):
            return "LEFT"
        return "RIGHT"

    my_head = tuple(my_trail[-1]) if my_trail else (1, 2)
    opp_head = tuple(opp_trail[-1]) if opp_trail else (BOARD_W - 2, BOARD_H - 3)

    # occupancy by my trail and opp trail
    occ_my = [[0.0] * BOARD_W for _ in range(BOARD_H)]
    occ_opp = [[0.0] * BOARD_W for _ in range(BOARD_H)]
    for (x, y) in my_trail:
        if 0 <= y < BOARD_H and 0 <= x < BOARD_W:
            occ_my[y][x] = 1.0
    for (x, y) in opp_trail:
        if 0 <= y < BOARD_H and 0 <= x < BOARD_W:
            occ_opp[y][x] = 1.0

    feats: List[float] = []
    # flatten channels row-major
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            feats.append(occ_my[y][x])
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            feats.append(occ_opp[y][x])

    # dir one-hot
    feats += _one_hot(4, DIR_TO_IDX.get(current_dir(my_trail), 3))
    feats += _one_hot(4, DIR_TO_IDX.get(current_dir(opp_trail), 3))

    # boosts normalized
    my_boosts = state.get("agent1_boosts", 3) if player_number == 1 else state.get("agent2_boosts", 3)
    opp_boosts = state.get("agent2_boosts", 3) if player_number == 1 else state.get("agent1_boosts", 3)
    feats.append(float(my_boosts) / 3.0)
    feats.append(float(opp_boosts) / 3.0)

    # turbn count normalized (cap at 500 in docs; engine uses 200 here)
    turn = float(state.get("turn_count", 0))
    feats.append(min(turn, 500.0) / 500.0)

    # moves around head (cheap): count of empty moves ignoring wrap trails beyond 1-step
    def neighbors(head):
        x, y = head
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx = (x + dx) % BOARD_W
            ny = (y + dy) % BOARD_H
            yield (nx, ny)

    occ_all = set(tuple(p) for p in my_trail) | set(tuple(p) for p in opp_trail)
    safe = 0
    for n in neighbors(my_head):
        if n not in occ_all:
            safe += 1
    feats.append(float(safe) / 4.0)

    # two-step safe estimate (greedy expansion)
    two_step = 0
    for n in neighbors(my_head):
        if n in occ_all:
            continue
        local_safe = 0
        for n2 in neighbors(n):
            if n2 not in occ_all:
                local_safe += 1
        # count a branch as safe if it has at least 1 follow-up
        if local_safe > 0:
            two_step += 1
    feats.append(float(two_step) / 4.0)

    # Wrapped distances
    dx = _wrap_dist(my_head[0], opp_head[0], BOARD_W)
    dy = _wrap_dist(my_head[1], opp_head[1], BOARD_H)
    feats.append(dx / float(BOARD_W))
    feats.append(dy / float(BOARD_H))

    # flood-fill reachable area (simple BFS ignoring opponent simultaneity)
    def flood_fill(start, blocked):
        if start in blocked:
            return 0
        seen = {start}
        dq = [(start[0], start[1])]
        size = 0
        while dq:
            x, y = dq.pop()
            size += 1
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx = (x + dx) % BOARD_W
                ny = (y + dy) % BOARD_H
                nb = (nx, ny)
                if nb in blocked or nb in seen:
                    continue
                seen.add(nb)
                dq.append(nb)
        return size

    blocked = occ_all  # both trails are walls
    my_area = flood_fill(my_head, blocked)
    opp_area = flood_fill(opp_head, blocked)
    total_cells = float(BOARD_W * BOARD_H)
    feats.append(my_area / total_cells)
    feats.append(opp_area / total_cells)

    # Voronoi advantage (approximate by BFS wavefront distances)
    def voronoi_advantage(my_h, opp_h, blocked):
        from_me = {my_h: 0}
        from_opp = {opp_h: 0}
        q_me = [my_h]
        q_opp = [opp_h]
        while q_me or q_opp:
            if q_me:
                cx, cy = q_me.pop(0)
                dm = from_me[(cx, cy)] + 1
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = (cx + dx) % BOARD_W, (cy + dy) % BOARD_H
                    if (nx, ny) in blocked or (nx, ny) in from_me:
                        continue
                    from_me[(nx, ny)] = dm
                    q_me.append((nx, ny))
            if q_opp:
                cx, cy = q_opp.pop(0)
                do = from_opp[(cx, cy)] + 1
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = (cx + dx) % BOARD_W, (cy + dy) % BOARD_H
                    if (nx, ny) in blocked or (nx, ny) in from_opp:
                        continue
                    from_opp[(nx, ny)] = do
                    q_opp.append((nx, ny))
        owned_me = 0
        owned_opp = 0
        all_cells = set(from_me.keys()) | set(from_opp.keys())
        for c in all_cells:
            dm = from_me.get(c, 10**9)
            do = from_opp.get(c, 10**9)
            if dm < do:
                owned_me += 1
            elif do < dm:
                owned_opp += 1
        return owned_me - owned_opp

    vor_adv = voronoi_advantage(my_head, opp_head, blocked)
    feats.append(vor_adv / total_cells)

    # Ray distances to nearest obstacle in 4 directions from heads
    def ray_dist(head, dx, dy):
        x, y = head
        dist = 0
        while True:
            x = (x + dx) % BOARD_W
            y = (y + dy) % BOARD_H
            dist += 1
            if (x, y) in blocked or dist >= max(BOARD_W, BOARD_H):
                break
        return dist

    # For my head
    feats.append(ray_dist(my_head, 0, -1) / float(BOARD_H))  # up
    feats.append(ray_dist(my_head, 0, 1) / float(BOARD_H))   # down
    feats.append(ray_dist(my_head, -1, 0) / float(BOARD_W))  # left
    feats.append(ray_dist(my_head, 1, 0) / float(BOARD_W))   # right
    # For opponent head
    feats.append(ray_dist(opp_head, 0, -1) / float(BOARD_H))
    feats.append(ray_dist(opp_head, 0, 1) / float(BOARD_H))
    feats.append(ray_dist(opp_head, -1, 0) / float(BOARD_W))
    feats.append(ray_dist(opp_head, 1, 0) / float(BOARD_W))

    return torch.tensor(feats, dtype=torch.float32)


class DecisionNet(nn.Module):
    """Two-layer MLP with slightly larger width and dropout.
    Outputs direction logits (4) and a boost logit (1).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 384)
        self.fc2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(p=0.15)
        self.head_dir = nn.Linear(192, 4)
        self.head_boost = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        dir_logits = self.head_dir(x)
        boost_logit = self.head_boost(x).squeeze(-1)
        return dir_logits, boost_logit


def load_model(weights_path: str | None) -> Tuple[DecisionNet, int]:
    # determine feature size by building a dummy state
    dummy = {
        "board": [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)],
        "agent1_trail": [(1, 2), (2, 2)],
        "agent2_trail": [(17, 15), (16, 15)],
        "agent1_boosts": 3, "agent2_boosts": 3, "turn_count": 0,
    }
    feat = build_features(dummy, player_number=1)
    input_dim = feat.numel()

    model = DecisionNet(input_dim)

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            # Allow training with newer feature sets without crashing at load
            pass
    model.eval()
    torch.set_num_threads(1)
    return model, input_dim
