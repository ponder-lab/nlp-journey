# coding=utf-8
# created by msg on 2020/6/30
import config
import torch
from torch.nn import CrossEntropyLoss
from util import logger


def loss_fn(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    """
    logits = outputs[0]

    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，
    # shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(config.DEVICE)

    loss_func = CrossEntropyLoss(ignore_index=config.PAD_ID, reduction="sum")
    loss = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(config.PAD_ID)
    num_targets = not_ignore.long().sum().item()
    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def train_fn(model, data_loader, optimizer, scheduler, epoch, batch_num, multi_gpu):
    logger.info("start training model...")
    # 手动开启训练，默认是预测
    model.train()
    # 用于统计每次梯度累计的loss 和 准确率
    running_loss = 0
    running_accuracy = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录 out of memory的次数
    oom_time = 0

    for batch_idx, input_ids in enumerate(data_loader):
        input_ids = input_ids.to(config.DEVICE)
        try:
            outputs = model(input_ids=input_ids)
            loss, accuracy = loss_fn(outputs, labels=input_ids)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if config.GRADIENT_ACCUMULATION > 1:
                loss = loss / config.GRADIENT_ACCUMULATION
                accuracy = accuracy / config.GRADIENT_ACCUMULATION
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            # 反向计算梯度
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION == 0:
                # 更新权重参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                # 更新学习率
                scheduler.step()
                overall_step += 1
                # 更新日志信息
                if overall_step % config.LOG_STEP == 0:
                    logger.info("batch {}/{} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, batch_num,
                                                                                       epoch + 1, running_loss, running_accuracy))
                running_loss = 0
                running_accuracy = 0
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logger.info("WARNING: gpu ran out of gpu memory,times: {}".format(oom_time))
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
    all_accuracy = 0
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(data_loader):
            input_ids.to(config.DEVICE)
            outputs = model(input_ids=input_ids)
            loss, accuracy = loss_fn(outputs, labels=input_ids)

            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            logger.info("evaluate batch {}/{}, loss: {}, accuracy: {}".format(batch_idx, test_batch_num, loss, accuracy))
            all_loss += loss
            all_accuracy += accuracy

        logger.info("finished evaluating")
    mean_loss = all_loss / test_batch_num
    mean_accuracy = all_accuracy / test_batch_num
    return mean_loss, mean_accuracy
