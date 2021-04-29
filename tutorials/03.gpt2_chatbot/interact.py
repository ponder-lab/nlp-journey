# coding=utf-8
# created by msg on 2020/7/1

import os
from datetime import datetime

import torch
import torch.nn.functional as F

import config
from model import create_model
from util import logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def run():
    logger.info('using device:{}'.format(config.DEVICE))

    # 对话model
    model, _ = create_model(pre_trained=True)
    model.to(config.DEVICE)
    model.eval()

    if not os.path.exists(config.SAVE_SAMPLES_PATH):
        os.makedirs(config.SAVE_SAMPLES_PATH)
    sample_path = os.path.join(config.SAVE_SAMPLES_PATH, "samples.txt")
    samples_file = open(sample_path, "a", encoding="utf8")
    samples_file.write("聊天记录: {}\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和 chatbot 聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user: ")
            samples_file.write("user: {}\n".format(text))

            history.append(config.TOKENIZER.encode(text))
            input_ids = [config.TOKENIZER.cls_token_id]

            for history_id, history_utter in enumerate(history[-config.MAX_HISTORY_LEN:]):
                input_ids.extend(history_utter)
                input_ids.append(config.TOKENIZER.sep_token_id)
            curr_input_tensor = torch.tensor(input_ids).long().to(config.DEVICE)
            generated = []
            # 最多生成max_len个token
            for _ in range(config.MAX_LEN):
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for index in set(generated):
                    next_token_logits[index] /= config.REPETITION_PENALTY
                next_token_logits /= config.TEMPERATURE

                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[config.TOKENIZER.convert_tokens_to_ids("[UNK]")] = -float("Inf")
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=config.TOP_K, top_p=config.TOP_P)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # 判断是否有response生成了[SEP],将已生成了[SEP]的response进行标记
                if next_token == config.TOKENIZER.sep_token_id:
                    break
                generated.append(next_token.item())
                # 将新生成的token与原来的token进行拼接
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

            history.append(generated)
            text = config.TOKENIZER.convert_ids_to_tokens(generated)
            print("chat bot:" + "".join(text))

            if config.SAVE_SAMPLES_PATH:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if config.SAVE_SAMPLES_PATH:
                samples_file.close()
            break


if __name__ == "__main__":
    run()
