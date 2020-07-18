# coding=utf-8
# created by msg on 2020/6/30
import config
import transformers
from transformers.modeling_gpt2 import GPT2LMHeadModel


def create_model(pre_trained=False):
    if pre_trained:
        model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH)
    else:
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(config.CONFIG_JSON_FILE)
        model = GPT2LMHeadModel(config=model_config)
    # model.resize_token_embeddings(vocab_size)
    n_ctx = model.config.to_dict().get("n_ctx")
    return model, n_ctx
