# coding=utf-8
# created by msg on 2020/6/19

import config
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from util import logger
from collections import Counter
from matplotlib.pyplot import MultipleLocator


def generate_subset():
    """
    生成训练子集
    """
    with open(config.RAW_DATA_PATH, "r", encoding="utf8") as f:
        data = f.read()
        dialogues = data.split("\n\n")
        subset_size = min(len(dialogues), config.SUBSET_SIZE)

    subset_path = os.path.join(config.SUBSET_DATA_PATH, "train_{}w.txt".format(int(subset_size / 10000)))
    with open(subset_path, "w", encoding="utf8") as f:
        logger.info("generating {} subset, please wait a few seconds. ".format(subset_size))
        for dialogue_index, dialogue in tqdm(enumerate(dialogues), total=len(dialogues)):
            if dialogue_index >= subset_size:
                break
            for utterance in dialogue.split("\n"):
                f.writelines(utterance + "\n")
            f.writelines("\n")


def compute_dialogue_length():
    """
    查看聊天语料中的dialogue的长度分布
    """
    with open(config.RAW_DATA_PATH, "r", encoding="utf8") as f:
        data = f.read()
    dialogues = data.split("\n\n")
    dialogues_lengths = [len(dialogue.replace("\n", "")) for dialogue in dialogues]
    counter = Counter(dialogues_lengths)
    dialogue_length_arr = list(counter)

    num_arr = [counter[element] for element in list(counter)]
    print(counter[300])

    x_major_locator = MultipleLocator(100)  # MultipleLocator用于设置刻度间隔
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xlabel("dialogue length")
    plt.ylabel("number of dialogue")

    plt.scatter(dialogue_length_arr, num_arr)

    plt.show()


if __name__ == "__main__":
    generate_subset()
    # compute_dialogue_length()
