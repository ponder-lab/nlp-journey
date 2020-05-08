import os
from collections import Counter

import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from smartnlp.custom.layer.attention import VanillaRNNAttention
from smartnlp.utils.loader import load_bin_word2vec
from smartnlp.utils.loader import load_en_stopwords, save_config, load_config, load_model, save_model
from smartnlp.utils.plot_model_history import plot


class BasicTextClassifier:
    """
    A basic text classifier.
    Argument:
        model_path: The model path: if you dont have a model, after the training,
        this will be the path to save your model.
        config_path: The path to save or load some configurations to speed up a little bit.
        train: Whether you want to train the model or just load model from the disk.
        train_file_path: If train is True, the file path must not be None.
        vector_path: If you want to use a trained word2vec model, just set this.
    """

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 train_file_path=None,
                 vector_path=None):
        self.model_path = model_path
        self.config_path = config_path
        if not train:
            assert config_path is not None, 'The config path cannot be None.'
            config = load_config(self.config_path)
            if not config:
                (self.word_index, self.max_len, self.embeddings) = config
                self.model = load_model(self.model_path, self.build_model())
            if not self.model:
                print('The model cannot be loaded：', self.model_path)
        else:
            self.vector_path = vector_path
            self.train_file_path = train_file_path
            self.x_train, self.y_train, self.x_test, self.y_test, self.word_index, self.max_index = self.load_data()
            self.max_len = self.x_train.shape[1]
            config = load_config(self.config_path)
            if not config:
                self.embeddings = load_bin_word2vec(self.word_index, self.vector_path, self.max_index)
                save_config((self.word_index, self.max_len, self.embeddings), self.config_path)
            else:
                (_, _, self.embeddings) = config
            self.model = self.train()
            save_model(self.model, self.model_path)

    # 全连接的一个简单的网络, 仅用来作为基类测试代码通过，速度快, 但是分类效果特别差
    def build_model(self):
        inputs = Input(shape=(self.max_len,))

        x = Embedding(len(self.embeddings),
                      300,
                      weights=[self.embeddings],
                      trainable=False)(inputs)

        x = Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, batch_size=512, epochs=20):
        model = self.build_model()
        # early_stop配合checkpoint使用，可以得到val_loss最小的模型
        early_stop = EarlyStopping(patience=3, verbose=1)
        checkpoint = ModelCheckpoint(os.path.join(self.model_path, 'weights.{epoch:03d}-{val_loss:.3f}.h5'),
                                     verbose=1,
                                     monitor='val_loss',
                                     save_best_only=True)
        history = model.fit(self.x_train,
                            self.y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpoint, early_stop],
                            validation_data=(self.x_test, self.y_test))
        plot(history)
        return model

    def predict(self, text):
        indices = None
        if isinstance(text, str):
            indices = [[self.word_index[t] if t in self.word_index.keys() else 0 for t in text.split()]]
        elif isinstance(text, list):
            indices = [[self.word_index[t] if t in self.word_index.keys() else 0 for t in tx.split()] for tx in text]
        if indices:
            indices = pad_sequences(indices, 500)
            return self.model.predict(indices)
        else:
            return []

    # 默认选用keras自带的处理好的数据来做模拟分类
    def load_data(self):
        return self.load_data_from_keras()

    @staticmethod
    def load_data_from_keras(max_len=500):
        (x_train, y_train), (x_test, y_test) = imdb.load_data()

        word_index = imdb.get_word_index()

        x_train = pad_sequences(x_train, maxlen=max_len)
        x_test = pad_sequences(x_test, maxlen=x_train.shape[1])
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        max_index = max([max(x) for x in x_train])

        return x_train, y_train, x_test, y_test, word_index, max_index

    # 用自己的数据集做训练（格式：分好词的句子##标签，如：我 很 喜欢 这部 电影#pos）
    def load_data_from_scratch(self, test_size=0.2, max_len=100):
        assert self.train_file_path is not None, 'file must not be none '
        stopwords = load_en_stopwords()
        with open(self.train_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split('##') for line in lines]
        x = [line[0] for line in lines]
        x = [line.split() for line in x]
        data = [word for xx in x for word in xx]
        y = [line[0] for line in lines]

        counter = Counter(data)
        vocab = [k for k, v in counter.items() if v >= 5]

        word_index = {k: v for v, k in enumerate(vocab)}

        max_sentence_length = max([len(words) for words in x])
        max_len = max_len if max_sentence_length > max_len else max_sentence_length

        x_data = [[word_index[word] for word in words if word in word_index.keys() and word not in stopwords] for words
                  in x]
        x_data = pad_sequences(x_data, maxlen=max_len)

        y_data = to_categorical(y)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
        return x_train, y_train, x_test, y_test, word_index


# cnn
class TextCnnClassifier(BasicTextClassifier):

    def __init__(self, model_path,
                 config_path,
                 train=False,
                 vector_path=None,
                 filter_sizes=None,
                 num_filters=256,
                 drop=0.5):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5, 6]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.drop = drop
        super(TextCnnClassifier, self).__init__(model_path=model_path,
                                                config_path=config_path,
                                                train=train,
                                                vector_path=vector_path)

    def build_model(self, input_shape=(500,)):
        inputs = Input(shape=input_shape, dtype='int32')
        embedding = Embedding(self.max_index + 1,
                              300,
                              weights=[self.embeddings],
                              trainable=False)(inputs)
        filter_results = []
        for i, filter_size in enumerate(self.filter_sizes):
            c = Conv1D(self.num_filters,
                       kernel_size=filter_size,
                       padding='valid',
                       activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(0.001),
                       name='conv-' + str(i + 1))(embedding)
            max_pool = GlobalMaxPooling1D(name='max-pool-' + str(i + 1))(c)
            filter_results.append(max_pool)
        concat = Concatenate()(filter_results)
        dropout = Dropout(self.drop)(concat)
        output = Dense(units=1,
                       activation='sigmoid',
                       name='dense')(dropout)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model


# han: Hierarchical Attention Networks
class TextHanClassifier(BasicTextClassifier):
    # 对长文本比较好, 可以在长文本中截断处理，把一段作为一个sentence
    def build_model(self):
        # word part
        input_word = Input(shape=(int(self.max_len / 5),))
        x_word = Embedding(len(self.embeddings),
                           300,
                           weights=[self.embeddings],
                           trainable=False)(input_word)
        x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)
        x_word = VanillaRNNAttention(256)(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        inputs = Input(shape=(self.max_len,))  # (5, self.max_len) ：(篇章最多包含的句子，每句包含的最大词数)
        reshape = Reshape((5, int(self.max_len / 5)))(inputs)
        x_sentence = TimeDistributed(model_word)(reshape)
        x_sentence = Bidirectional(LSTM(128, return_sequences=True))(x_sentence)
        x_sentence = VanillaRNNAttention(256)(x_sentence)

        output = Dense(1, activation='sigmoid')(x_sentence)
        model = Model(inputs=inputs, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, batch_size=128, epochs=2):
        # 比较耗费资源，笔记本GPU跑不动，只好减小batch_size
        return super(TextHanClassifier, self).train(batch_size=batch_size, epochs=epochs)


# rnn + cnn
class TextRCNNClassifier(BasicTextClassifier):

    def build_model(self):
        inputs = Input((self.max_len,))
        embedding = Embedding(len(self.embeddings),
                              300,
                              weights=[self.embeddings],
                              trainable=False)(inputs)
        x_context = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])
        cs = []
        for kernel_size in range(1, 5):
            c = Conv1D(128, kernel_size, activation='relu')(x)
            cs.append(c)
        pools = [GlobalAveragePooling1D()(c) for c in cs] + [GlobalMaxPooling1D()(c) for c in cs]
        x = Concatenate()(pools)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, batch_size=128, epochs=2):
        super(TextRCNNClassifier, self).train(batch_size=batch_size, epochs=epochs)


class TextRnnClassifier(BasicTextClassifier):
    """
    rnn
    """

    def __init__(self, model_path, config_path, train, vector_path):
        super(TextRnnClassifier, self).__init__(model_path=model_path,
                                                config_path=config_path,
                                                train=train,
                                                vector_path=vector_path)

    def build_model(self):
        inputs = Input(shape=(self.max_len,))
        x = Embedding(len(self.embeddings),
                      300,
                      weights=[self.embeddings],
                      trainable=False)(inputs)
        x = Bidirectional(LSTM(150))(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
        y = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class TextRNNAttentionClassifier(BasicTextClassifier):
    """
    rnn + attention
    """

    def build_model(self):
        inputs = Input(shape=(self.max_len,))
        output = Embedding(len(self.embeddings),
                           300,
                           weights=[self.embeddings],
                           trainable=False)(inputs)
        output = Bidirectional(LSTM(150,
                                    return_sequences=True,
                                    dropout=0.25,
                                    recurrent_dropout=0.25))(output)
        output = VanillaRNNAttention(300)(output)
        output = Dense(128, activation="relu")(output)
        output = Dropout(0.25)(output)
        output = Dense(1, activation="sigmoid")(output)
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, batch_size=128, epochs=5):
        super(TextRNNAttentionClassifier, self).train(batch_size=batch_size, epochs=epochs)
