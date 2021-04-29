# coding=utf-8
# created by msg on 2020/7/1
import os
import config
import torch
from dataset import DialogueDataset
from engine import train_fn, eval_fn
from preprocess import process_raw_data
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from util import logger
from model import create_model


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    """
    input_len = [len(btc) for btc in batch]
    max_input_len = max(input_len)
    input_ids = [
        btc + ([config.PAD_ID] * (max_input_len - btc_len))
        for (btc, btc_len) in zip(batch, input_len)
    ]
    return torch.tensor(input_ids, dtype=torch.long)


def run():
    logger.info("using device: {}".format(config.DEVICE))
    train_data = process_raw_data()
    train_list, test_list = train_test_split(train_data,
                                             test_size=0.2,
                                             random_state=34)

    # 加载GPT2模型
    model, n_ctx = create_model(False)
    model.to(config.DEVICE)
    # 是否使用多块GPU进行并行运算: 可以选择要使用哪几块显卡来进行训练
    multi_gpu = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Using more than one GPUs to train...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE_NUM
        model = DataParallel(
            model, device_ids=[int(i) for i in config.DEVICE_NUM.split(",")])
        multi_gpu = True

    # 记录模型参数数量
    num_parameters = sum(
        [parameter.numel() for parameter in model.parameters()])
    logger.info("number of model parameters: {}".format(num_parameters))

    # 加载数据
    logger.info("loading training data")
    train_dataset = DialogueDataset(train_list, n_ctx)
    batch_num = len(train_dataset) // config.BATCH_SIZE
    test_dataset = DialogueDataset(test_list, n_ctx)
    test_batch_num = len(test_dataset) // config.BATCH_SIZE

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=collate_fn)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=1,
                                  collate_fn=collate_fn)

    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(
        len(train_data_loader) * config.EPOCHS / config.BATCH_SIZE /
        config.GRADIENT_ACCUMULATION)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARM_STEPS,
        num_training_steps=total_steps)

    logger.info("start training...")
    best_loss = 100
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_fn(model, train_data_loader, optimizer, scheduler, epoch,
                 batch_num, multi_gpu)
        loss, accuracy = eval_fn(model, test_data_loader, test_batch_num, multi_gpu)
        if loss < best_loss or accuracy > best_accuracy:
            logger.info('saving model for epoch {}, best loss: {}'.format(
                epoch + 1, loss))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(config.MODEL_PATH)
            best_loss = loss
            best_accuracy = accuracy


if __name__ == '__main__':
    run()
