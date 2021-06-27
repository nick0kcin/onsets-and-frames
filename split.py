from pathlib import Path

import pandas as pd
from random import random


def merge_into_single_dataframe(groups):
    merge = None
    for group in groups:
        if merge is not None:
            merge = merge.append(group[1], ignore_index=True)
        else:
            merge = group[1]
    return merge

if __name__ == "__main__":
    with open("mp3list.txt", "w") as f:
        for path in Path("./files").glob("*"):
            f.write(str(path) + "\n")
    data = pd.read_csv("data2.csv")
    data = data[data["diff"] < 2]
    data["author"] = list(map(lambda x: x.partition("+")[0], data["midi"]))
    groups = [group for group in data.groupby("author")]
    sorted_groups = sorted(groups, key=lambda g: g[1].shape[0], reverse=True)
    train, val, test = [], [], []
    for group in sorted_groups:
        r = random()
        if r < 0.035:
            val.append(group)
        elif r > 0.965:
            test.append(group)
        else:
            train.append(group)
    merge_into_single_dataframe(train).to_csv("train.csv", index=None)
    merge_into_single_dataframe(val).to_csv("val.csv", index=None)
    merge_into_single_dataframe(test).to_csv("test.csv", index=None)