import os

import hydra
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from model.lit_model import LitSASRec
from utils.data import data_partition, download_data
from utils.data_loader import SASRecDataset
from utils.plot_metrics import plot_mlflow_metrics

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    download_data()
    seed_everything(42)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # подготовка данных
    dataset = data_partition(cfg.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset

    cfg.usernum = usernum
    cfg.itemnum = itemnum

    OmegaConf.save(config=cfg, f=os.path.join(save_dir, "meta.yaml"))

    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        f.write(f"usernum: {usernum}\n")
        f.write(f"itemnum: {itemnum}\n")

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for k, v in OmegaConf.to_container(cfg).items():
            f.write(f"{k},{v}\n")

    model = LitSASRec(cfg=cfg, dataset=dataset)

    train_data = SASRecDataset(user_train, itemnum, cfg.maxlen)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    avg_len = sum(len(v) for v in user_train.values()) / len(user_train)
    print("Average sequence length: %.2f" % avg_len)

    logger = MLFlowLogger(
        experiment_name=cfg.dataset,
        run_name=cfg.train_dir,
        tracking_uri="file:./mlruns",
    )

    run_id = logger.run_id
    print(f"[MLflow] Run ID: {run_id}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="last",
        save_last=True,
        save_top_k=1,
        monitor="valid_NDCG10",
        mode="max",
    )

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,
        accelerator="auto",
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
        ],
    )

    trainer.fit(model, train_loader)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.dataset)

    runs = client.search_runs(
        experiment.experiment_id,
        filter_string=f"tags.mlflow.runName = '{cfg.train_dir}'",
        order_by=["start_time DESC"],
    )
    run_id = runs[0].info.run_id

    plot_mlflow_metrics(run_id)


if __name__ == "__main__":
    main()
