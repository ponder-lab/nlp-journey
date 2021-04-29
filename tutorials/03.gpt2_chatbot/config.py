# coding=utf-8
# created by msgi on 2020/6/4
import torch
from transformers import BertTokenizer

# 模型训练相关路径
CONFIG_JSON_FILE = "../config/model_config_dialogue_small.json"
RAW_DATA_PATH = "../input/train.txt"
VOCAB_FILE = "../input/vocab_small.txt"

# 模型输出相关路径
LOG_PATH = "../log/"
MODEL_PATH = "../models"
SAVE_SAMPLES_PATH = "../samples/"

# 训练相关参数
SEED = 34
BATCH_SIZE = 8
EPOCHS = 10
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1.5e-4
WARM_STEPS = 2000
MAX_GRAD_NORM = 1.0
LOG_STEP = 1

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TOKENIZER = BertTokenizer(vocab_file=VOCAB_FILE)
PAD_ID = TOKENIZER.convert_tokens_to_ids("[PAD]")
DEVICE_NUM = "0,1"  # 设置使用哪些显卡

# 生成对话相关参数
MAX_HISTORY_LEN = 5
MAX_LEN = 25
REPETITION_PENALTY = 1.0
TEMPERATURE = 1
TOP_K = 8
TOP_P = 0
