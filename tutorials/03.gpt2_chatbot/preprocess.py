import config


def process_raw_data():
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，
    对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    with open(config.RAW_DATA_PATH, "rb") as f:
        data = f.read().decode("utf-8").strip()
        if "\r\n" in data:
            train_data = [
                dialogue.replace("\r\n", " [SEP] ")
                for dialogue in data.split("\r\n\r\n")
            ]
        else:
            train_data = [
                dialogue.replace("\n", " [SEP] ")
                for dialogue in data.split("\n\n")
            ]
    return train_data[:100]


if __name__ == "__main__":
    process_raw_data()
