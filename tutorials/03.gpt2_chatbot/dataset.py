# coding=utf-8
# created by msg on 2020/6/19
import config
import torch
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    def __init__(self, dataset, n_ctx):
        self.dataset = dataset
        self.tokenizer = config.TOKENIZER
        self.max_len = n_ctx

    def __getitem__(self, item):
        text = self.dataset[item]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )
        return inputs["input_ids"]

    def __len__(self):
        return len(self.dataset)
