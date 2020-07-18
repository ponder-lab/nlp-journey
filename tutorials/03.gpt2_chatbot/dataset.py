# coding=utf-8
# created by msg on 2020/6/19
import config
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    def __init__(self, dataset, n_ctx):
        self.dataset = dataset
        self.tokenizer = config.TOKENIZER
        self.max_len = n_ctx

    def __getitem__(self, item):
        text = self.dataset[item]
        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_len
        )
        return input_ids

    def __len__(self):
        return len(self.dataset)
