# -*- coding: utf-8 -*-

import os
from PIL import Image
import random
import numpy as np
import pandas as pd
import scikitplot
import seaborn as sns
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint

def DataSet():
    train_path_yawn ='G:\\Study\\fatiguedetector\\Dataset\\images\\3\\'
    train_path_sleepy ='G:\\Study\\fatiguedetector\\Dataset\\images\\2\\'
    train_path_nonsleepy = 'G:\\Study\\fatiguedetector\\Dataset\\images\\1\\'
    imglist_train_yawn = os. listdir(train_path_yawn)
    imglist_train_sleepy = os. listdir(train_path_sleepy)
    imglist_train_nonsleepy = os.listdir(train_path_nonsleepy)
    #Read files
    
    X_train = np.empty((len(imglist_train_sleepy) + len(imglist_train_nonsleepy) + len(imglist_train_yawn), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_sleepy) + len(imglist_train_nonsleepy) + len(imglist_train_yawn), 3))
    #Create numpy object, has 3 labels
    
    count1 = 0
    # count each image increase 1
    
    for img_name in imglist_train_sleepy:
        img_path = train_path_sleepy + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count1] = img
        Y_train[count1] = np.array((0,0,1))
        count1+=1
        
    for img_name in imglist_train_nonsleepy:
    
        img_path = train_path_nonsleepy + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count1] = img
        Y_train[count1] = np.array((1,0,0))
        count1+=1
        
    for img_name in imglist_train_yawn:

        img_path = train_path_yawn + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count1] = img
        Y_train[count1] = np.array((0,1,0))
        count1+=1        
        
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
   
    return X_train,Y_train

X_train,Y_train = DataSet()

np.save('G:\\Study\\fatiguedetector\\Dataset\\X_train_s', X_train, allow_pickle=True, fix_imports=True)
np.save('G:\\Study\\fatiguedetector\\Dataset\\Y_train_s', Y_train, allow_pickle=True, fix_imports=True)
#save dataset

print('X_train_S shape : ',X_train.shape)
print('y_train_S shape : ',Y_train.shape)

X_train_S, X_valid_S, y_train_S, y_valid_S = train_test_split(
  X_train,
  Y_train,
  shuffle=True, 
  stratify=Y_train,
  test_size=0.25, 
  random_state=42
)

model = ResNet50(weights=None,classes=3)

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=3)])

# # 保存模型路径
filepath='G:\\Study\\fatiguedetector\\Dataset\\model_s\\myresnet50model_3classes_times=1_e30bs12_etimes={epoch:02d}_valacc={val_accuracy:.2f}.h5'

# # # 需要按命名规则修改文件名（第一处需要修改的地方）：classes times epoches batchsize etimes valacc
# # # 需要自己手动按命名规则创建文件夹来存放模型

# 第一种保存方式：有一次提升, 则保存一次.
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max')
#callbacks_list = [checkpoint]
 
# 第二种保存方式：保存每一次（我采用的方法）
checkpoint = ModelCheckpoint(filepath, verbose=1,save_best_only=False,save_weights_only=False)
callbacks_list = [checkpoint]

# # train
#history=model.fit(X_train_S, y_train_S, epochs=30, batch_size=12,validation_split=0.25,callbacks=callbacks_list)
history=model.fit(X_train_S, y_train_S, epochs=30, batch_size=12,validation_data=(X_valid_S, y_valid_S),callbacks=callbacks_list)

# # #可以修改的参数（第二个需要改的地方）：epoches batchsize validation_split
# # evaluate
# # 第三种保存方式：仅保存最终模型
#model.save('C:\\Users\\Administrator\\Desktop\\test3\\myresnet50model_3classes_times=1_e30bs1_{epoch:02d}etimes_valacc={val_acc:.2f}.h5')
# # # 按命名规则修改（第三个需要改的地方）：classes times epoches batchsize etimes valacc

# # draw 并保存图片到文件夹下（手动另存为）
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
miou = history.history['mean_io_u']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, miou, 'r', label='Mean IoU')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, miou, 'r', label='Mean IoU')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('epoch_history_dcnn.png')
plt.show()

df_accu = pd.DataFrame({'train': history.history['accuracy'], 'valid': history.history['val_accuracy']})
df_loss = pd.DataFrame({'train': history.history['loss'], 'valid': history.history['val_loss']})

fig = plt.figure(0, (14, 4))
ax = plt.subplot(1, 2, 1)
sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
plt.title('Loss')
plt.tight_layout()

plt.savefig('performance_dist.png')
plt.show()

model = tf.keras.models.load_model('G:\\Study\\fatiguedetector\\Dataset\\model_s\\myresnet50model_3classes_times=1_e30bs12_etimes=29_valacc=0.69.h5')
yhat_valid = model.predict(X_valid_S)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid_S, axis=1), np.argmax(yhat_valid, axis=1), figsize=(7,7))
plt.savefig("confusion_matrix_dcnn.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid_S, axis=1) != np.argmax(yhat_valid, axis=1))}\n\n')
print(classification_report(np.argmax(y_valid_S, axis=1), np.argmax(yhat_valid, axis=1)))
