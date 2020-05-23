import transformers
import torch

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "./input/bert_base_chinese/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "./input/sentiment.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

