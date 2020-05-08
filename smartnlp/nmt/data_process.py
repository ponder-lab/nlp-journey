# coding=utf-8
# created by msgi on 2020/4/29 4:11 下午
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


class DataProcessor:

    def __init__(self, test_size=0.1, csv_path=None,
                 max_length=40, feature_col='', target_col='', buffer_size=20000, batch_size=64):
        self.test_size = test_size
        self.csv_path = csv_path
        self.feature_col = feature_col
        self.target_col = target_col
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def load_data(self):
        df = pd.read_csv(self.csv_path).dropna(inplace=True)
        return df

    def train_test_split(self, df):
        test_rows = int(len(df) * self.test_size)
        test = df.sample(test_rows)
        train = df[~df.isin(test)]

        train.dropna(inplace=True)
        test.dropna(inplace=True)

        return train, test

    def tokenizer(self, train, vocab_size=2 ** 13):
        if self.feature_col is None:
            feature_col = train.columns[0]
        else:
            feature_col = self.feature_col

        if self.target_col is None:
            target_col = train.columns[1]
        else:
            target_col = self.target_col
        global tokenizer_feat
        global tokenizer_tar
        tokenizer_tar = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line for line in train[target_col].values), target_vocab_size=vocab_size
        )

        tokenizer_feat = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line for line in train[feature_col].values), target_vocab_size=vocab_size
        )

        return tokenizer_feat, tokenizer_tar

    @staticmethod
    def encode(lang1, lang2):
        lang1 = [tokenizer_feat.vocab_size] + tokenizer_feat.encode(
            lang1.numpy()) + [tokenizer_feat.vocab_size + 1]

        lang2 = [tokenizer_tar.vocab_size] + tokenizer_tar.encode(
            lang2.numpy()) + [tokenizer_tar.vocab_size + 1]

        return lang1, lang2

    def filter_max_length(self, x, y):
        # 为了使本示例较小且相对较快，删除长度大于 max_length 个标记的样本。
        return tf.logical_and(tf.size(x) <= self.max_length,
                              tf.size(y) <= self.max_length)

    def tf_encode(self, feature, target):
        return tf.py_function(self.encode, [feature, target], [tf.int64, tf.int64])

    def to_tensor_dataset(self, data):
        return tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(data[self.feature_col].values, tf.string),
                tf.cast(data[self.target_col].values, tf.string)
            )
        )

    def init_preprocess(self, train_examples, val_examples):
        global tokenizer_tar
        global tokenizer_feat
        tokenizer_tar = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

        tokenizer_feat = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

        train_dataset = train_examples.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        # 将数据集缓存到内存中以加快读取速度。
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(self.buffer_size).padded_batch(self.batch_size,
                                                                             padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = val_examples.map(self.tf_encode)
        val_dataset = val_dataset.filter(self.filter_max_length).padded_batch(self.batch_size,
                                                                              padded_shapes=([-1], [-1]))
        return train_dataset, val_dataset, tokenizer_feat, tokenizer_tar

    def preprocess(self, train: pd.DataFrame, test: pd.DataFrame):
        train_data = self.to_tensor_dataset(train)
        test_data = self.to_tensor_dataset(test)

        train_dataset = train_data.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(self.buffer_size).padded_batch(self.batch_size,
                                                                             padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = test_data.map(self.tf_encode)
        test_dataset = test_dataset.filter(self.filter_max_length).padded_batch(self.batch_size,
                                                                                padded_shapes=([-1], [-1]))

        return train_dataset, test_dataset
