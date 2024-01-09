# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint

log_dir='G:\\Study\\fatiguedetector\\Dataset\\log'
file_writer_cm = tf.summary.create_file_writer(log_dir,filename_suffix='cm')
def plot_to_image(figure,logd_ir,epoch):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    fig=figure
    fig.savefig(os.path.join(log_dir,'during_train_confusion_epoch_{}.png'.format(epoch)))
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize = (8, 8))
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis], decimals = 2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm.iloc[i, j] > threshold.iloc[i] else "black"
        plt.text(j, i, cm.iloc[i, j], horizontalalignment = "center", color = color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(test_x)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = confusion_matrix(test_y, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure,log_dir,epoch)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# # model
X_train = np.load('G:\\Study\\fatiguedetector\\Dataset\\X_train_s.npy')
Y_train = np.load('G:\\Study\\fatiguedetector\\Dataset\\Y_train_s.npy')

model = ResNet50(weights=None,classes=3)

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # 保存模型路径
filepath='G:\\Study\\fatiguedetector\\Dataset\\model_s\\myresnet50model_3classes_times=1_e30bs12_etimes={epoch:02d}_valacc={val_accuracy:.2f}.h5'

# # # 需要按命名规则修改文件名（第一处需要修改的地方）：classes times epoches batchsize etimes valacc
# # # 需要自己手动按命名规则创建文件夹来存放模型

# 第一种保存方式：有一次提升, 则保存一次.
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max')
#callbacks_list = [checkpoint]
 
# 第二种保存方式：保存每一次（我采用的方法）
checkpoint = ModelCheckpoint(filepath, verbose=1,save_best_only=False,save_weights_only=False)
callbacks_list = [checkpoint, cm_callback]

# # train
history=model.fit(X_train, Y_train, epochs=30, batch_size=12,validation_split=0.25,callbacks=callbacks_list)

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
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
