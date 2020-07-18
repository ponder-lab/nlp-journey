# coding=utf-8
# created by msg on 2020/6/30
import config
import torch
from util import logger
from torch.utils.tensorboard import SummaryWriter


def train_fn(model, data_loader, optimizer, scheduler, epoch, batch_num,
             multi_gpu):
    logger.info("start training model...")
    # 手动开启训练，默认是预测
    model.train()
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 记录 out of memory的次数
    oom_time = 0

    summary_writer = SummaryWriter(log_dir=config.LOG_PATH)

    for batch_idx, input_ids in enumerate(data_loader):
        input_ids = input_ids.to(config.DEVICE)
        try:
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs[0]
            if multi_gpu:
                loss = loss.mean()
            if config.GRADIENT_ACCUMULATION > 1:
                loss = loss / config.GRADIENT_ACCUMULATION
            running_loss += loss.item()
            # 反向计算梯度
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.MAX_GRAD_NORM)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION == 0:
                # 更新权重参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                # 更新学习率
                scheduler.step()
                # 更新日志信息
                if (batch_idx + 1) % config.LOG_STEP == 0:
                    logger.info("batch: {}/{} of epoch: {}, loss: {}".format(
                        batch_idx + 1, batch_num, epoch + 1, running_loss))
                    summary_writer.add_scalar("loss", loss.item(), epoch * batch_num + batch_idx)
                running_loss = 0
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logger.info(
                    "WARNING: out of gpu memory,times: {}".format(oom_time))
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise Exception
    logger.info("epoch {} training finished".format(epoch + 1))


def eval_fn(model, data_loader, test_batch_num, multi_gpu):
    logger.info("starting evaluating model...")
    # 手动开启预测
    model.eval()

    all_loss = 0
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(data_loader):
            input_ids.to(config.DEVICE)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs[0]

            if multi_gpu:
                loss = loss.mean()
            logger.info("evaluate batch {}/{}, loss: {}".format(
                batch_idx, test_batch_num, loss))
            all_loss += loss.item()

        logger.info("finished evaluating")
    mean_loss = all_loss / test_batch_num
    return mean_loss
