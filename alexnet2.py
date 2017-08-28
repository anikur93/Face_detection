#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:44:21 2017

@author: sreehari
"""
import numpy as np
import matplotlib.image as mpimg
import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from tensorflow.contrib.layers import flatten

from alexnet_conv import AlexNet

X_train = []
y_train = labels

for i in range(len(images)):
    img = images[i].resize((120, 120))
    X_train.append(img)
    
for i in range(len(X_train)):
    X_train[i] = np.array(X_train[i])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#X_train = []
#
le.fit(y_train)
le.classes_
y_train = le.transform(y_train)
#le.inverse_transform(y_train)

le.transform(le.classes_)

#label_sets
#le.inverse_transform(y_val)[10]

label_sets = list(set(labels))
#y_train = labels

nb_classes = len(label_sets)
epochs = 10
batch_size = 256


    


X_train, y_train = shuffle(X_train, y_train)


#def rgb2gray(rgb):
#
#    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#    #gray = (r+g+b)/3
#    gray = 0.21 * r + 0.72 * g + 0.07 * b
#    gray = gray.reshape(32,32,1)
#    return gray
#
#def conversion(list1):
#    l = []
#    for i in range(len(list1)):
#        img = list1[i] 
#        gray = rgb2gray(img)
#        l.append(gray)
#    return l


def normalise(list1):
    l = []
    for i in range(len(list1)):
        img = list1[i]
        norm = (img - 128)/128
        l.append(norm)
    return l

#x_gray = conversion(X_train)
#x_nor_gray = normalise(x_gray) 

#with open('./train.p', 'rb') as f:
#    data = pickle.load(f)
x_train = normalise(X_train)
X_train = x_train

plt.imshow(X_train[0])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.13, random_state=0)

features = tf.placeholder(tf.float32, (None, 120, 120, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
conv4 = AlexNet(resized, feature_extract=True)

mu = 0
sigma = 0.1

fc0   = flatten(conv4)
##13*13*384 = 64896
##64896 -- 40000
fc1_W  = tf.Variable(tf.truncated_normal(shape=(64896, nb_classes), mean = mu, stddev = sigma))
fc1_b  = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc0, fc1_W, fc1_b)


#fc7 = tf.stop_gradient(fc7)
#shape = (fc7.get_shape().as_list()[-1], nb_classes)
#fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
#fc8b = tf.Variable(tf.zeros(nb_classes))
#logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)



cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc1_W, fc1_b])
init_op = tf.global_variables_initializer()

#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))



def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, len(X), batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * len(X_batch))
        total_acc += (acc * len(X_batch))

    return total_loss/len(X), total_acc/len(X)

save_file = './model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, len(X_train), batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
    saver.save(sess, save_file)
    print('Trained Model Saved.')

        

rn  = np.random.random_integers(0,len(X_val))        
ll = X_val[rn].reshape(1,120,120,3)
plt.imshow(X_val[rn])
     
#init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver.restore(sess, save_file)

    #sess.run(init)
    #print(le.transform([sess.run(preds, feed_dict={features :ll})]))    
    print(le.inverse_transform(sess.run(preds, feed_dict={features :ll})))
    print((sess.run(preds, feed_dict={features :ll})), y_val[rn])
    


    
import cv2
import face_recognition
from PIL import Image

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)  
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)   
    #i =i +1
    #if len(face_recognition.face_encodings(frame)) >= 1:       
    #     unknown_face_encoding = face_recognition.face_encodings(frame)
     #    i = 0 
     
    for face_location in face_locations:
            # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        ne_img = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 5)
        cropped_img = frame[top:bottom, left:right]
        
        ll = Image.fromarray(cropped_img)
        ll = ll.resize((120, 120))
        ll = np.array(ll)
        ll = ll.reshape(1,120,120,3)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver.restore(sess, save_file)
    #print(le.transform([sess.run(preds, feed_dict={features :ll})]))    
            name=le.inverse_transform(sess.run(preds, feed_dict={features :ll}))
        cv2.putText(ne_img, name[0],(left, top), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow('preview',ne_img)
        
            #plt.imshow(ne_img) xywh y:y+h x:x+w
            #unknown_face_encoding_present = unknown_face_encoding[i]
            #import face_recognition
            #results = face_recognition.compare_faces(known_faces, unknown_face_encoding_present)
            #print(results)
            #i = i +1
            
          #  for i in range(len(results)):
           #     if results[i] == True:
            #        cv2.putText(ne_img,img_dict[i+1],(right,top), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                    
        
        
        # Display the resulting frame
        
    #cv2.imshow('preview', cropped_img)
    #cv2.imwrite('akshay'+str(i)+'.png',cropped_img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()