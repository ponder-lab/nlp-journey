import time

import flask
import torch
import torch.nn as nn
from flask import Flask, request

import config
from model import BERTBaseUncased

app = Flask(__name__)

MODEL = None


def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LEN
    review = str(sentence)

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_length
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_length - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(config.DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(config.DEVICE, dtype=torch.long)
    mask = mask.to(config.DEVICE, dtype=torch.long)

    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'sentence': str(sentence),
        'time_taken': str(time.time() - start_time)
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))
    MODEL.to(config.DEVICE)
    MODEL.eval()

    app.run()
