import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from dataset import SASRecDataset
from model.lit_model import LitSASRec
from model.sasrec import SASRecModel
from utils import data_partition, download_data

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    download_data()
    seed_everything(42)
    save_dir = f"{cfg.dataset}_{cfg.train_dir}"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for k, v in OmegaConf.to_container(cfg).items():
            f.write(f"{k},{v}\n")

    dataset = data_partition(cfg.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset

    train_data = SASRecDataset(user_train, itemnum, cfg.maxlen)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    avg_len = sum(len(v) for v in user_train.values()) / len(user_train)
    print("average sequence length: %.2f" % avg_len)

    base_model = SASRecModel(usernum, itemnum, cfg)
    model = LitSASRec(base_model, cfg, dataset)

    logger = MLFlowLogger(
        experiment_name=cfg.dataset,
        run_name=cfg.train_dir,
        tracking_uri="file:./mlruns",
    )

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,
        accelerator="auto",
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
