import pandas as pd
from PIL import Image
import numpy as np
import os
import pickle


def to_array(file):  # 输入一个地址
    image = Image.open(file)  # 读取图片
    im_array = np.array(image)  # 转换成array
    return im_array


df_train = pd.DataFrame({'img': [], 'label': []})  # 创建DataFrame
df_test = pd.DataFrame({'img': [], 'label': []})

for i in range(10):
    for path in os.listdir('./MNIST_Dataset/train_images/' + str(i)):
        img_file = './MNIST_Dataset/train_images/' + str(i) + '/' + path
        df_train = df_train.append(pd.DataFrame({'img': [to_array(img_file)], 'label': [int(i)]}))

for path in os.listdir('./MNIST_Dataset/test_images/'):
    img_file = './MNIST_Dataset/test_images/' + path
    df_test = df_test.append(pd.DataFrame({'img': [to_array(img_file)], 'label': [int(path[0])]}))

df_train['label'] = df_train['label'].astype("int")  # 转换label列为int
df_test['label'] = df_test['label'].astype("int")

''' csv的格式保存列表不方便速度又慢
df_train.to_csv('./save/train.csv')
df_test.to_csv('./save/test.csv')
'''

with open('save/train.pickle', 'wb')as f:  # save
    pickle.dump(df_train, f)
with open('save/test.pickle', 'wb')as f:
    pickle.dump(df_test, f)

print("save over!")
