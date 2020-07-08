# coding=utf-8
# created by msg on 2020/6/20

import copy
import os
from datetime import datetime

import config
import torch
import torch.nn.functional as F
from util import logger
from model import create_model


def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float("Inf")):
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # safety check
    if top_k > 0:
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def run():
    logger.info('using device:{}'.format(config.DEVICE))

    # 对话model
    dialogue_model, _ = create_model(pre_trained=True)
    dialogue_model.to(config.DEVICE)
    dialogue_model.eval()

    # 互信息mmi model
    mmi_model, _ = create_model(pre_trained=True, mmi=True)
    mmi_model.to(config.DEVICE)
    mmi_model.eval()

    if not os.path.exists(config.SAVE_SAMPLES_PATH):
        os.makedirs(config.SAVE_SAMPLES_PATH)
    samples_file = open(config.SAVE_SAMPLES_PATH + "/mmi_samples.txt", "a", encoding="utf8")
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

            # 用于批量生成response，维度为(batch_size, token_len)
            input_ids = [copy.deepcopy(input_ids) for _ in range(config.BATCH_SIZE)]
            curr_input_tensors = torch.tensor(input_ids).long().to(config.DEVICE)

            # 二维数组，维度为(生成的response的最大长度，batch_size)，
            # generated[i,j]表示第j个response的第i个token的id
            generated = []

            # 标记是否所有response均已生成结束，若第i个response生成结束，
            # 即生成了sep_token_id，则将i放入finish_set
            finish_set = set()
            # 最多生成max_len个token
            for _ in range(config.MAX_LEN):
                outputs = dialogue_model(input_ids=curr_input_tensors)
                next_token_logits = outputs[0][:, -1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for index in range(config.BATCH_SIZE):
                    for token_id in set([token_ids[index] for token_ids in generated]):
                        next_token_logits[index][token_id] /= config.REPETITION_PENALTY
                next_token_logits = next_token_logits / config.TEMPERATURE

                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                for next_token_logit in next_token_logits:
                    next_token_logit[config.TOKENIZER.convert_tokens_to_ids("[UNK]")] = -float("Inf")
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=config.TOP_K, top_p=config.TOP_P)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # 判断是否有response生成了[SEP],将已生成了[SEP]的response进行标记
                for index, token_id in enumerate(next_token[:, 0]):
                    if token_id == config.TOKENIZER.sep_token_id:
                        finish_set.add(index)
                # 检验是否所有的response均已生成[SEP]
                finish_flag = True  # 是否所有的response均已生成[SEP]的token
                for index in range(config.BATCH_SIZE):
                    if index not in finish_set:  # response批量生成未完成
                        finish_flag = False
                        break
                if finish_flag:
                    break
                generated.append([token.item() for token in next_token[:, 0]])
                # 将新生成的token与原来的token进行拼接
                curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=-1)
            candidate_responses = []  # 生成的所有候选response
            for batch_index in range(config.BATCH_SIZE):
                response = []
                for token_index in range(len(generated)):
                    if generated[token_index][batch_index] != config.TOKENIZER.sep_token_id:
                        response.append(generated[token_index][batch_index])
                    else:
                        break
                candidate_responses.append(response)

            # mmi模型的输入
            if config.DEBUG:
                print("candidate response:")
            samples_file.write("candidate response:\n")

            min_loss = float("Inf")
            best_response = ""
            for response in candidate_responses:
                mmi_input_id = [config.TOKENIZER.cls_token_id]  # 每个input以[CLS]为开头
                mmi_input_id.extend(response)
                mmi_input_id.append(config.TOKENIZER.sep_token_id)
                for history_utter in reversed(history[-config.MAX_HISTORY_LEN:]):
                    mmi_input_id.extend(history_utter)
                    mmi_input_id.append(config.TOKENIZER.sep_token_id)
                mmi_input_tensor = torch.tensor(mmi_input_id).long().to(config.DEVICE)
                out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)

                loss = out[0].item()
                if config.DEBUG:
                    text = config.TOKENIZER.convert_ids_to_tokens(response)
                    print("{} loss:{}".format("".join(text), loss))
                samples_file.write("{} loss:{}\n".format("".join(text), loss))

                if loss < min_loss:
                    best_response = response
                    min_loss = loss
            history.append(best_response)
            text = config.TOKENIZER.convert_ids_to_tokens(best_response)
            print("chatbot:" + "".join(text))

            if config.SAVE_SAMPLES_PATH:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if config.SAVE_SAMPLES_PATH:
                samples_file.close()
            break


if __name__ == "__main__":
    run()
