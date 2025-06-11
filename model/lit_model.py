import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from dataset import SASRecDataset
from evaluate import evaluate, evaluate_valid


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
