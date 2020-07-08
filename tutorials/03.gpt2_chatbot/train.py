# coding=utf-8
# created by msg on 2020/7/1
import os
import random

import config
import numpy as np
import torch
import transformers
from dataset import DialogueDataset
from engine import *
from preprocess import *
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from util import logger
from datetime import datetime
from model import create_model


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    input_len = [len(btc) for btc in batch]
    max_input_len = max(input_len)
    input_ids = [btc + ([config.PAD_ID] * (max_input_len - btc_len)) for (btc, btc_len) in zip(batch, input_len)]
    return torch.tensor(input_ids, dtype=torch.long)


def run():
    logger.info("using device: {}".format(config.DEVICE))

    if config.TRAIN_MMI:
        train_data = process_mmi_raw_data()
    else:
        train_data = process_raw_data()
    train_list, test_list = train_test_split(train_data, test_size=0.2, random_state=34)

    # 加载GPT2模型
    model, n_ctx = create_model(mmi=config.TRAIN_MMI)
    model.to(config.DEVICE)

    # 是否使用多块GPU进行并行运算: 可以选择要使用哪几块显卡来进行训练
    multi_gpu = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Using GPU to train...")
        model = DataParallel(model, device_ids=[int(i) for i in config.DEVICE_NUM.split(",")])
        multi_gpu = True
    else:
        logger.info("Using cpu to train...")

    # 记录模型参数数量
    num_parameters = sum([parameter.numel() for parameter in model.parameters()])
    logger.info("number of model parameters: {}".format(num_parameters))

    # 加载数据
    logger.info("loading training data")
    train_dataset = DialogueDataset(train_list, n_ctx)
    batch_num = len(train_dataset) // config.BATCH_SIZE
    test_dataset = DialogueDataset(test_list, n_ctx)
    test_batch_num = len(test_dataset) // config.BATCH_SIZE

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TEST_NUM_WORKERS,
        collate_fn=collate_fn
    )

    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(len(train_data_loader) * config.EPOCHS / config.BATCH_SIZE / config.GRADIENT_ACCUMULATION)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        correct_bias=True
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARM_STEPS,
        num_training_steps=total_steps
    )

    logger.info("start training...")
    best_accuracy = 0
    best_loss = 100
    for epoch in range(config.EPOCHS):
        epoch_start_time = datetime.now()
        train_fn(model, train_data_loader, optimizer, scheduler, epoch, batch_num, multi_gpu)
        logger.info("time for epoch {}: {}".format(epoch + 1, datetime.now() - epoch_start_time))
        loss, accuracy = eval_fn(model, test_data_loader, test_batch_num, multi_gpu)
        if accuracy > best_accuracy or loss < best_loss:
            logger.info('saving model for epoch {}, best accuracy is {}'.format(epoch + 1, accuracy))
            if config.TRAIN_MMI:  # 当前训练MMI模型
                model_path = config.MMI_MODEL_PATH
            else:
                model_path = config.DIALOGUE_MODEL_PATH
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            best_accuracy = accuracy
            best_loss = loss


if __name__ == '__main__':
    run()
