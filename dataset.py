import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SASRecDataset(Dataset):
    def __init__(self, user_train, itemnum, maxlen):
        self.user_train = user_train
        self.user_ids = list(user_train.keys())
        self.itemnum = itemnum
        self.maxlen = maxlen

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user = self.user_ids[index]
        items = self.user_train[user]
        seq = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)
        neg = np.zeros([self.maxlen], dtype=np.int64)

        nxt = items[-1]
        idx = self.maxlen - 1
        ts = set(items)

        for i in reversed(items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = self.sample_neg(ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (
            torch.LongTensor(seq),
            torch.LongTensor(pos),
            torch.LongTensor(neg),
        )

    def sample_neg(self, positives):
        t = random.randint(1, self.itemnum)
        while t in positives:
            t = random.randint(1, self.itemnum)
        return t
