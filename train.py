import os

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from dataset import SASRecDataset
from evaluate import evaluate, evaluate_valid
from model.sasrec import SASRecModel
from utils import data_partition, download_data

torch.set_float32_matmul_precision("high")


class LitSASRec(LightningModule):
    def __init__(self, model, cfg, dataset):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.dataset = dataset

    def forward(self, seq):
        return self.model(seq)

    def training_step(self, batch, batch_idx):
        seq, pos, neg = batch
        seq_emb = self.model(seq)
        pos_logits, neg_logits = self.model.predict(seq_emb, pos, neg)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits, pos_labels, reduction="none"
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits, neg_labels, reduction="none"
        )

        istarget = (pos != 0).float().view(-1)
        loss = (loss_pos + loss_neg) * istarget
        loss = torch.sum(loss) / torch.sum(istarget)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def val_dataloader(self):
        user_train, user_valid, _, usernum, itemnum = self.dataset
        val_data = SASRecDataset(user_train, itemnum, self.cfg.maxlen)
        return DataLoader(
            val_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        valid_scores = evaluate_valid(
            self.model,
            self.dataset,
            self.cfg,
            self.device,
        )
        test_scores = evaluate(
            self.model,
            self.dataset,
            self.cfg,
            self.device,
        )

        self.log("valid_NDCG10", valid_scores[0], prog_bar=True)
        self.log("valid_HR10", valid_scores[1], prog_bar=True)
        self.log("test_NDCG10", test_scores[0])
        self.log("test_HR10", test_scores[1])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=(0.9, 0.98),
        )


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
