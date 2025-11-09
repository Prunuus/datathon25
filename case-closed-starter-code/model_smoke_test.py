import os
import json
import torch
from model import build_features, load_model

# Dummy state mirroring local tester format
state = {
    "board": [[0 for _ in range(20)] for _ in range(18)],
    "agent1_trail": [(1, 2), (2, 2)],
    "agent2_trail": [(17, 15), (16, 15)],
    "agent1_length": 2,
    "agent2_length": 2,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 1,
    "player_number": 1,
}

model_path = os.environ.get("MODEL_WEIGHTS")
model, input_dim = load_model(model_path)
feat = build_features(state, player_number=1)
print("Feature dim:", feat.numel())

if model is None:
    print("No model weights loaded. Integration will fall back to heuristic.")
else:
    with torch.no_grad():
        dir_logits, boost_logit = model(feat.unsqueeze(0))
        print("dir_logits:", dir_logits.tolist())
        print("boost_logit:", boost_logit.item())
