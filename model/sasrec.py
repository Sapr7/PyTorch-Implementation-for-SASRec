import torch
import torch.nn as nn

from model.blocks import PositionalEncoding, SASRecBlock


class SASRecModel(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super().__init__()
        self.item_emb = nn.Embedding(
            itemnum + 1,
            args.hidden_units,
            padding_idx=0,
        )
        self.pos_enc = PositionalEncoding(args.hidden_units, args.maxlen)
        self.dropout = nn.Dropout(args.dropout_rate)

        self.blocks = nn.ModuleList(
            [
                SASRecBlock(
                    args.hidden_units,
                    args.num_heads,
                    args.hidden_units,
                    args.dropout_rate,
                )
                for _ in range(args.num_blocks)
            ]
        )
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen

    def forward(self, input_seq):
        mask = (input_seq != 0).float().unsqueeze(-1)
        seq_emb = self.item_emb(input_seq)
        seq_emb = self.pos_enc(seq_emb)
        seq_emb = self.dropout(seq_emb)
        seq_emb *= mask

        key_padding_mask = input_seq == 0  # [B, T]

        for block in self.blocks:
            seq_emb = block(seq_emb, padding_mask=key_padding_mask)

        return nn.functional.layer_norm(seq_emb, [self.hidden_units])

    def predict(self, seq_emb, pos_ids, neg_ids):
        B, T, D = seq_emb.size()
        seq_flat = seq_emb.view(B * T, D)
        pos_emb = self.item_emb(pos_ids.view(-1))
        neg_emb = self.item_emb(neg_ids.view(-1))

        pos_logits = torch.sum(seq_flat * pos_emb, dim=-1)
        neg_logits = torch.sum(seq_flat * neg_emb, dim=-1)
        return pos_logits, neg_logits

    def test_step(self, seq_emb, candidates):
        final_state = seq_emb[:, -1, :]
        candidate_emb = self.item_emb(candidates)
        return torch.bmm(candidate_emb, final_state.unsqueeze(-1)).squeeze(-1)
