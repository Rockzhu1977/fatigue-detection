# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance as dist
from imutils import face_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
import argparse
import imutils
import time
import dlib
import cv2
import math
import os
from PIL import Image
import random
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

mapper = {
    0: "nodrowsy",
    1: "drowsy",
    2: "yawn",
}

X_train = np.load('G:\\Study\\fatiguedetector\\Dataset\\X_train_s.npy')
Y_train = np.load('G:\\Study\\fatiguedetector\\Dataset\\Y_train_s.npy')

X_train_S, X_valid_S, y_train_S, y_valid_S = train_test_split(
  X_train,
  Y_train,
  shuffle=True, 
  stratify=Y_train,
  test_size=0.25, 
  random_state=42
)

model = load_model('G:\\Study\\fatiguedetector\\Dataset\\model_s\\myresnet50model_3classes_times=1_e30bs12_etimes=30_valacc=0.84.h5')
#model = tf.keras.models.load_model('G:\\Study\\fatiguedetector\\Dataset\\model_s\\myresnet50model_3classes_times=1_e30bs12_etimes=30_valacc=0.84.h5')
yhat_valid = model.predict(X_valid_S)

print(len(X_valid_S))
print(len(yhat_valid))
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid_S, axis=1), np.argmax(yhat_valid, axis=1), figsize=(7,7))
plt.savefig("confusion_matrix_dcnn.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid_S, axis=1) != np.argmax(yhat_valid, axis=1))}\n\n')
print(classification_report(np.argmax(y_valid_S, axis=1), np.argmax(yhat_valid, axis=1)))


# # test
img_path = "G:\\Study\\fatiguedetector\\Dataset\\train_data\\images\\2\\009_sleepyCombination_451_drowsy.jpg"
img = image.load_img(img_path, target_size=(224, 224))


plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)

predict = model.predict(img)
print(predict)
predict=np.argmax(predict,axis=1)
print(predict)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('G:\\Study\\IT9501AppliedProject\\demo\\fatigue_detecting\\model\\shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

image = cv2.imread(img_path) #assuming image1.png is located in the same directory as this script
image = imutils.resize(image, width=500)

#if the image is colour
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# show the face number
	cv2.putText(image, "Face: {}".format(mapper[predict[0]]), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	#for (x, y) in shape:
	#	cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

