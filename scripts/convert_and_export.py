import os

import hydra
import torch
from omegaconf import DictConfig

from model.lit_model import LitSASRec
from utils.export import export_to_onnx, export_to_trt, verify_onnx


@hydra.main(config_path="configs", config_name="infer", version_base="1.1")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    checkpoint_path = os.path.join(cfg.save_dir, "last.ckpt")

    with open(os.path.join(cfg.save_dir, "info.txt")) as f:
        for line in f:
            k, v = line.strip().split(":")
            setattr(cfg, k, int(v))

    model = LitSASRec.load_from_checkpoint(
        checkpoint_path=checkpoint_path, cfg=cfg, dataset=None
    ).model

    model.to(device)
    model.eval()

    onnx_path = os.path.join(cfg.save_dir, "sasrec.onnx")
    export_to_onnx(model, input_size=(1, cfg.maxlen), save_path=onnx_path)
    verify_onnx(onnx_path)

    trt_path = os.path.join(cfg.save_dir, "sasrec.trt")
    export_to_trt(onnx_path, trt_path)


if __name__ == "__main__":
    main()
