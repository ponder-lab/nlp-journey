# coding=utf-8
# created by msgi on 2020/6/4
import torch
from transformers import BertTokenizer

# 模型训练相关路径
CONFIG_JSON_FILE = "../models/config/model_config_dialogue_small.json"
RAW_DATA_PATH = "../input/data/train_0w.txt"
VOCAB_FILE = "../input/data/vocab_small.txt"
SUBSET_DATA_PATH = "../input/data/"

# 模型输出相关路径
LOG_PATH = "../log/training.log"
DIALOGUE_MODEL_PATH = "../models/dialogue_model"
MMI_MODEL_PATH = "../models/mmi_model"
SAVE_SAMPLES_PATH = "../models/samples/"

# 训练相关参数
SUBSET_SIZE = 100
TRAIN_MMI = True
DEBUG = True
SEED = 34
BATCH_SIZE = 8
EPOCHS = 10
TRAIN_NUM_WORKERS = 4
TEST_NUM_WORKERS = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1.5e-4
WARM_STEPS = 2000
MAX_GRAD_NORM = 1.0
LOG_STEP = 1

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TOKENIZER = BertTokenizer(vocab_file=VOCAB_FILE)
PAD = '[PAD]'
PAD_ID = TOKENIZER.convert_tokens_to_ids(PAD)
DEVICE_NUM = "0,1"  # 设置使用哪些显卡

# 生成对话相关参数
MAX_HISTORY_LEN = 5
MAX_LEN = 25
REPETITION_PENALTY = 1.0
TEMPERATURE = 1
TOP_K = 8
TOP_P = 0
