import config
from util import logger
from tqdm import tqdm


def process_raw_data(reverse=False):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，
    对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param reverse: 是否倒序
    """
    with open(config.RAW_DATA_PATH, "rb") as f:
        data = f.read().decode("utf8").strip()
        if "\r\n" in data:
            train_data = [dialogue.replace("\r\n", " [SEP] ") for dialogue in data.split("\r\n\r\n")]
        else:
            train_data = [dialogue.replace("\n", " [SEP] ") for dialogue in data.split("\n\n")]
        if reverse:
            train_data = [" [SEP] ".join(reversed(dialogue.split(" [SEP] "))) for dialogue in train_data]
    return train_data


def process_mmi_raw_data():
    """
    对原始语料进行处理，将原始语料的每段对话进行翻转，然后转换为用于train MMI模型的token id，
    对于每个dialogue，将其处于成如下形式"[CLS]utterance N[SEP]utterance N-1[SEP]utterance N-2[SEP]"
    """
    return process_raw_data(True)


if __name__ == "__main__":
    process_mmi_raw_data()
