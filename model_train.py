# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import random

def DataSet():
    train_path_yawn ='G:\\Study\\fatiguedetector\\Dataset\\train_data\\images\\3\\'
    train_path_sleepy ='G:\\Study\\fatiguedetector\\Dataset\\train_data\\images\\2\\'
    train_path_nonsleepy = 'G:\\Study\\fatiguedetector\\Dataset\\train_data\\images\\1\\'
    imglist_train_yawn = os. listdir(train_path_yawn)
    imglist_train_sleepy = os. listdir(train_path_sleepy)
    imglist_train_nonsleepy = os.listdir(train_path_nonsleepy)
    #Read files
    
    X_train = np.empty((len(imglist_train_sleepy) + len(imglist_train_nonsleepy) + len(imglist_train_yawn), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_sleepy) + len(imglist_train_nonsleepy) + len(imglist_train_yawn), 3))
    #Create numpy object, has 3 labels
    
    count1 = 0
    # count each image increase 1
    
    # read all images from 
    for img_name in imglist_train_sleepy:
        img_path = train_path_sleepy + img_name
        # 通过 image.load_img() 函数读取对应的图片，并转换成目标大小
        #  image 是 tensorflow.keras.preprocessing 中的一个对象
        img = image.load_img(img_path, target_size=(224, 224))
        # 将图片转换成 numpy 数组，并除以 255 ，归一化
        # 转换之后 img 的 shape 是 （224，224，3）
        img = image.img_to_array(img) / 255.0
        # 将处理好的图片装进定义好的 X_train 对象中
        X_train[count1] = img
        # 将对应的标签装进 Y_train 对象中，困的标签设为(0,0,1)
        Y_train[count1] = np.array((0,0,1))
        count1+=1    #计数器＋1
        
    # 遍历清醒状态的所有图片并写入训练集
    for img_name in imglist_train_nonsleepy:
    
        img_path = train_path_nonsleepy + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count1] = img
        Y_train[count1] = np.array((1,0,0))
        count1+=1
        
    #遍历有点困状态下所有图片并写入训练集
    for img_name in imglist_train_yawn:

        img_path = train_path_yawn + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count1] = img
        Y_train[count1] = np.array((0,1,0))
        count1+=1        
        
    # 不准备测试集，已经将所有图片写入训练集中
    
	# 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
   
    return X_train,Y_train    #Dataset()函数返回两个训练集

X_train,Y_train = DataSet()

np.save('G:\\Study\\fatiguedetector\\Dataset\\train_data\\X_train', X_train, allow_pickle=True, fix_imports=True)
np.save('G:\\Study\\fatiguedetector\\Dataset\\train_data\\Y_train', Y_train, allow_pickle=True, fix_imports=True)
#保存数据集

print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
