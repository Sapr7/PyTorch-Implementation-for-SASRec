import os
import subprocess
from collections import defaultdict


def download_data():
    if not os.path.exists("data"):
        print("Downloading data via DVC...")
        subprocess.run(["dvc", "pull", "data.dvc"], check=True)
    else:
        print("Data already exists, skipping download.")


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    with open(f"data/{fname}.txt", "r") as f:
        for line in f:
            u, i = map(int, line.strip().split())
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return user_train, user_valid, user_test, usernum, itemnum
