# coding=utf-8
# created by msgi on 2020/4/25 10:01 上午
import os

import numpy as np
import logging
import pickle
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from smartnlp.utils.basic_log import Log

log = Log(logging.INFO)


# 加载词向量(获取需要的词向量: 谷歌词向量)
def load_bin_word2vec(word_index, word2vec_path, max_index):
    log.info('Begin load word vector from bin...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embeddings = _select_vectors(word2vec, word_index, max_index)
    log.info('End load word vector from bin...')
    return embeddings


# 加载词向量(获取需要的词向量：自己训练好的文本词向量)
def load_text_vector(word_index, word2vec_path, max_index):
    log.info('Begin load word vector from text...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    embeddings = _select_vectors(word2vec, word_index, max_index)
    log.info('End load word vector from text...')
    return embeddings


def _select_vectors(word2vec, word_index, max_index):
    embedding_dim = word2vec.vector_size
    embeddings = 1 * np.random.randn(max_index + 1, embedding_dim)
    embeddings[0] = 0

    for word, index in word_index.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    return embeddings


# 加载英文停用词
def load_en_stopwords(file_path=None):
    return stopwords.words('english')


# 加载中文停用词
def load_zh_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


# 写入配置文件
def save_config(config, config_path):
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)


# 加载配置文件
def load_config(config_path):
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        config = None
    return config


# 保存模型
def save_model(model, model_path, weights_only=True):
    if model:
        if weights_only:
            model.save_weights(os.path.join(model_path, 'weights.h5'))
        else:
            model.save(os.path.join(model_path, 'model.h5'))


# 加载模型
def load_model(model_path, model=None, weights_only=True):
    try:
        if weights_only and model:
            model.load_weights(os.path.join(model_path, 'weights.h5'))
        else:
            model = load_model(os.path.join(model_path, 'model.h5'))
    except FileNotFoundError:
        model = None
    return model
