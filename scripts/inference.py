import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from model.lit_model import LitSASRec
from utils.io import load_user_item_dict


def predict_top_k(lit_model: LitSASRec, user_history, itemnum, cfg, device):
    seq = np.zeros([cfg.maxlen], dtype=np.int64)
    idx = cfg.maxlen - 1
    for item in reversed(user_history):
        if idx < 0:
            break
        seq[idx] = item
        idx -= 1

    rated = set(user_history)
    rated.add(0)
    candidates = []
    while len(candidates) < 100:
        t = np.random.randint(1, itemnum + 1)
        if t not in rated:
            candidates.append(t)

    seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
    item_tensor = torch.LongTensor(candidates).unsqueeze(0).to(device)

    seq_emb = lit_model(seq_tensor)
    logits = lit_model.model.test_step(seq_emb, item_tensor)
    scores = -logits[0].detach().cpu().numpy()
    return [x for _, x in sorted(zip(scores, candidates))][: cfg.top_k]


@hydra.main(config_path="../configs", config_name="infer", version_base="1.1")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_histories = load_user_item_dict(cfg.input_txt)

    with open(os.path.join(cfg.save_dir, "info.txt")) as f:
        meta = {line.split(":")[0]: int(line.split(":")[1]) for line in f}
    cfg.usernum = meta["usernum"]
    cfg.itemnum = meta["itemnum"]

    ckpt_path = os.path.join(cfg.save_dir, "last.ckpt")
    model = LitSASRec.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        dataset=None,
    )
    model.to(device)
    model.eval()

    results = {}
    for uid, history in user_histories.items():
        recs = predict_top_k(model, history, cfg.itemnum, cfg, device)
        results[uid] = recs

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    with open(cfg.output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
