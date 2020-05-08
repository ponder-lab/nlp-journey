import logging
import os
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_addons as ta
from tensorflow.keras.preprocessing.sequence import pad_sequences

from smartnlp.utils.basic_log import Log
from smartnlp.utils.loader import load_model, load_config, save_model, save_config

log = Log(logging.INFO)


class BiLSTMCRFModel(tf.keras.Model):
    def __init__(self, hidden_units, vocab_size, label_size, embedding_size):
        super(BiLSTMCRFModel, self).__init__()
        self.num_hidden = hidden_units
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = ta.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens


class BiLSTMCRFNamedEntityRecognition:
    def __init__(self,
                 model_path,
                 config_path,
                 embed_dim=200,
                 rnn_units=200,
                 epochs=1,
                 batch_size=64,
                 train=False,
                 file_path=None):
        self.model_path = model_path
        self.config_path = config_path
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        self.file_path = file_path
        self.batch_size = batch_size
        # 词性tag
        self.tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

        # 非训练模式，直接加载模型
        if not train:
            self.word2idx = load_config(self.config_path)
            self.model = load_model(self.model_path, self._build_model())
            assert self.model is not None, '训练模型无法获取'
        else:
            (self.train_x, self.train_y), (self.test_x, self.test_y), self.word2idx = self._load_data()
            self.epochs = epochs
            self.model = self.train()
        save_model(self.model, self.model_path)
        save_config(self.word2idx, self.config_path)

    # 训练
    def train(self):
        model = self._build_model()

        optimizer = tf.keras.optimizers.Adam(0.01)

        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt.restore(tf.train.latest_checkpoint(self.model_path))
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  self.model_path,
                                                  checkpoint_name='model.ckpt',
                                                  max_to_keep=3)
        best_acc = 0
        step = 0
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        train_dataset = train_dataset.shuffle(len(self.train_x)).batch(self.batch_size, drop_remainder=True)

        for epoch in range(self.epochs):
            for _, (text_batch, labels_batch) in enumerate(train_dataset):
                step = step + 1
                loss, logits, text_lens = self.train_one_step(model, optimizer, text_batch, labels_batch)
                if step % 20 == 0:
                    accuracy = self.get_acc_one_step(model, logits, text_lens, labels_batch)
                    log.info('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
                    if accuracy > best_acc:
                        best_acc = accuracy
                        ckpt_manager.save()
                        log.info("model saved")

        return model

    @staticmethod
    def train_one_step(model, optimizer, text_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits, text_lens

    @staticmethod
    def get_acc_one_step(model, logits, text_lens, labels_batch):
        paths = []
        accuracy = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            viterbi_path, _ = ta.text.viterbi_decode(logit[:text_len], model.transition_params)
            paths.append(viterbi_path)
            correct_prediction = tf.equal(
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                     dtype=tf.int32),
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                     dtype=tf.int32)
            )
            accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = accuracy / len(paths)
        return accuracy

    # 识别句子中的实体
    def predict(self, predict_text):
        # predict_text = ''
        sent, length = self._preprocess_data(predict_text)
        raw = self.model.predict(sent)[0][-length:]
        result = np.argmax(raw, axis=1)
        result_tags = [self.tags[i] for i in result]

        per, loc, org = '', '', ''
        for s, t in zip(predict_text, result_tags):
            if t in ('B-PER', 'I-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s
        results = ['person:' + per, 'location:' + loc, 'organization:' + org]
        print(results)
        return results

    # 预测的时候，进行数据处理转换
    def _preprocess_data(self, data, max_len=100):
        x = [self.word2idx.get(w[0].lower(), 1) for w in data]
        length = len(x)
        x = pad_sequences([x], max_len)
        return x, length

    # 构造模型
    def _build_model(self):
        model = BiLSTMCRFModel(self.rnn_units, len(self.word2idx), len(self.tags), self.embed_dim)
        return model

    # 训练数据预处理
    def _load_data(self):
        train = self._parse_data(os.path.join(self.file_path, 'train.data'))
        test = self._parse_data(os.path.join(self.file_path, 'test.data'))

        # 统计每个字出现的频次
        word_counts = Counter(row[0].lower() for sample in train for row in sample)
        vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
        word2idx = dict((w, i) for i, w in enumerate(vocab))

        train = self._process_data(train, word2idx, self.tags)
        test = self._process_data(test, word2idx, self.tags)
        return train, test, word2idx

    @staticmethod
    def _process_data(data, word2idx, chunk_tags, max_len=None):
        if max_len is None:
            max_len = max(len(s) for s in data)
        x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
        y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

        x = pad_sequences(x, max_len, padding='post')
        y_chunk = pad_sequences(y_chunk, max_len, padding='post')

        return x, y_chunk

    @staticmethod
    def _parse_data(file_path):
        with open(file_path, 'rb') as f:
            string = f.read().decode('utf-8')
            data = [[row.split() for row in sample.split('\n')] for sample in
                    string.strip().split('\n' + '\n')]
        return data


if __name__ == '__main__':
    biLSTM = BiLSTMCRFNamedEntityRecognition('model/ner/',
                                             'model/ner/config.pkl',
                                             train=True,
                                             file_path='/Users/msgi/workspace/pythons/nlp-journey/examples/data/ner')
