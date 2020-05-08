# coding=utf-8
# created by msgi on 2020/4/26 6:54 下午

import io
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from smartnlp.utils.plot_model_history import plot
from smartnlp.utils.loader import *


class VanillaEmbeddingModel:
    def __init__(self, model_path, embedding_dim, file_path=None, training=False):
        self.embedding_dim = embedding_dim
        self.file_path = file_path
        self.model_path = model_path
        self.train_batches, self.test_batches, self.encoder = self.preprocess()

        if training:
            self.model = self.train_model()
            save_model(self.model, self.model_path)
            self.save_embeddings_to_file()
        else:
            self.model = load_model(self.model_path, self._build_model())

    def _build_model(self):
        model = keras.Sequential([
            layers.Embedding(self.encoder.vocab_size, self.embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self):
        model = self._build_model()
        history = model.fit(
            self.train_batches,
            epochs=10,
            validation_data=self.test_batches, validation_steps=20)
        plot(history)
        return model

    def save_embeddings_to_file(self):
        e = self.model.layers[0]
        weights = e.get_weights()[0]
        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')

        for num, word in enumerate(self.encoder.subwords):
            vec = weights[num + 1]  # skip 0, it's padding.
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_v.close()
        out_m.close()

    @staticmethod
    def preprocess():
        (train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                                  split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                                  with_info=True, as_supervised=True)
        train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=([None], []))
        test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=([None], []))

        encoder = info.features['text'].encoder
        return train_batches, test_batches, encoder


if __name__ == '__main__':
    model = VanillaEmbeddingModel('model', 16, training=True)
    model.save_embeddings_to_file()
