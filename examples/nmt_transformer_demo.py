# coding=utf-8
# created by msgi on 2020/4/28 4:37 下午
import os
import tensorflow_datasets as tfds

import tensorflow as tf
from datetime import datetime
from smartnlp.nmt.training import Trainer
from smartnlp.custom.learning_rate.learning_rate import CustomSchedule
from smartnlp.custom.model.transformer import Transformer
from smartnlp.nmt.data_process import DataProcessor

# 设置使用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

data_processor = DataProcessor()
train_dataset, test_dataset, tokenizer_feat, tokenizer_tar = data_processor.init_preprocess(train_examples,
                                                                                            val_examples)

EPOCHS = 100
num_layers = 6
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_feat.vocab_size + 2
target_vocab_size = tokenizer_tar.vocab_size + 2
dropout_rate = 0.1

# Custom Scheduler
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Transformer
transformer = Transformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers,
                          target_vocab_size=target_vocab_size, input_vocab_size=input_vocab_size,
                          dff=dff, rate=dropout_rate)

# Trainer
print(f'\n\nBeginning training for {EPOCHS} epochs @ {datetime.now()}...\n')
trainer = Trainer(train_dataset=train_dataset,
                  test_dataset=test_dataset,
                  learning_rate=learning_rate,
                  optimizer=optimizer,
                  transformer=transformer,
                  epochs=EPOCHS,
                  checkpoint_path='./models/checkpoints/',
                  tb_log_dir='./logs/gradient_tape/'

                  )

loss_hist, acc_hist = trainer.train()
