# coding=utf-8
# created by msgi on 2020/5/27
import tensorflow as tf
from tensorflow.keras.layers import *


class VanillaClassificationModel(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim=300):
        super(VanillaClassificationModel, self).__init__()
        self.emb = Embedding(vocab_size, emb_dim)

        self.lamb = Lambda(lambda t: tf.reduce_mean(t, axis=1))
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(16, activation='relu')
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.emb(inputs)
        x = self.lamb(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.classifier(x)


class CNNClassificationModel(tf.keras.Model):
    def __init__(self, vocab_size, filter_sizes, emb_dim=300, num_filters=256, drop=0.5):
        super(CNNClassificationModel, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.drop = drop
        self.emb = Embedding(vocab_size, emb_dim)
        self.convolutions = [Conv1D(self.num_filters,
                                    kernel_size=filter_size,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))
                             for filter_size in self.filter_sizes]
        self.max_pool = GlobalMaxPooling1D()
        self.concat = Concatenate()
        self.dropout = Dropout(self.drop)
        self.classifier = Dense(units=1,
                                activation='sigmoid',
                                name='dense')

    def call(self, inputs):
        x = self.emb(inputs)
        filter_results = []
        for convolution in self.convolutions:
            x = convolution(x)
            max_pool = self.max_pool(x)
            filter_results.append(max_pool)
        concat = self.concat(filter_results)
        dropout = self.dropout(concat)
        return self.classifier(dropout)