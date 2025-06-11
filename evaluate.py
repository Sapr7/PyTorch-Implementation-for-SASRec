import copy
import random

import numpy as np
import torch


def evaluate(model, dataset, args, device):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = (
        random.sample(range(1, usernum + 1), 10000)
        if usernum > 10000
        else range(1, usernum + 1)
    )
    model.eval()

    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

            seq = np.zeros([args.maxlen], dtype=np.int64)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            while len(item_idx) < 101:
                t = np.random.randint(1, itemnum + 1)
                if t not in rated:
                    item_idx.append(t)

            seq_tensor = torch.LongTensor(seq)
            seq_tensor = seq_tensor.unsqueeze(0).to(device)
            item_tensor = torch.LongTensor(item_idx)
            item_tensor = item_tensor.unsqueeze(0).to(device)
            logits = model.test_step(model(seq_tensor), item_tensor)
            predictions = -logits[0].cpu().numpy()

            rank = predictions.argsort().argsort()[0]

            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

            if valid_user % 100 == 0:
                print(".", end="", flush=True)

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, device):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = (
        random.sample(range(1, usernum + 1), 10000)
        if usernum > 10000
        else range(1, usernum + 1)
    )
    model.eval()

    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue

            seq = np.zeros([args.maxlen], dtype=np.int64)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            while len(item_idx) < 101:
                t = np.random.randint(1, itemnum + 1)
                if t not in rated:
                    item_idx.append(t)

            seq_tensor = torch.LongTensor(seq)
            seq_tensor = seq_tensor.unsqueeze(0).to(device)
            item_tensor = torch.LongTensor(item_idx)
            item_tensor = item_tensor.unsqueeze(0).to(device)
            logits = model.test_step(model(seq_tensor), item_tensor)
            predictions = -logits[0].cpu().numpy()

            rank = predictions.argsort().argsort()[0]

            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

            if valid_user % 100 == 0:
                print(".", end="", flush=True)

    return NDCG / valid_user, HT / valid_user
