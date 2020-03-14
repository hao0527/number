import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


with open('save/train.pickle','rb')as f:    # restore
    df_train = pickle.load(f)
with open('save/test.pickle','rb')as f:
    df_test = pickle.load(f)
# 载入数据到元组
(train_image, train_label), (test_image, test_label) = (df_train.img, df_train.label),(df_test.img, df_test.label)
train_image = np.array([train_image], dtype='int')[0]
test_image = np.array([test_image], dtype='int')[0]    # 将shape转为(60000,28,28)


# 归一化
train_image = train_image/255
test_image = test_image/255

# 独热编码
train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)

# 创建第一个模型结构
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))    # 有十个输出
model.summary()

# 配置该模型的学习流程
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
              )

model.fit(train_image, train_label_onehot, epochs=5)

model.evaluate(test_image, test_label_onehot)

