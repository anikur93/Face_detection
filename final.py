from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
import pandas as pd

X_train = []
y_train = labels

for i in range(len(y_train)):
    if y_train[i] == 'anirudh':
        y_train[i] = 1
    if y_train[i] == 'vivek':
        y_train[i] = 2
    if y_train[i] == 'shyam':
        y_train[i] = 3
    if y_train[i] == 'tamaghna':
        y_train[i] = 4


#y_train = pd.DataFrame(labels)
#y_train = pd.get_dummies(y_train)

for i in range(len(images)):
    img = images[i].resize((32, 32))
    X_train.append(img)
    
for i in range(len(X_train)):
    X_train[i] = np.array(X_train[i])
    


from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import numpy as np
import matplotlib.image as mpimg
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #gray = (r+g+b)/3
    gray = 0.21 * r + 0.72 * g + 0.07 * b
    gray = gray.reshape(32,32,1)
    return gray

def conversion(list1):
    l = []
    for i in range(len(list1)):
        img = list1[i] 
        gray = rgb2gray(img)
        l.append(gray)
    return l


def normalise(list1):
    l = []
    for i in range(len(list1)):
        img = list1[i]
        norm = (img - 128)/128
        l.append(norm)
    return l

x_gray = conversion(X_train)
x_nor_gray = normalise(x_gray) 


###right -left =150, top-bottom=150
import tensorflow as tf

EPOCHS = 15
BATCH_SIZE = 158

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    #conv1 = tf.nn.dropout(conv1, keep_prob)

    ## 28x28x6 -- 24x24x16.
    # 28,28,6 -- 24,24,16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    
    #24,24,16 ---19,19,24
    conv3_W = tf.Variable(tf.truncated_normal(shape=(6, 6, 16, 24), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(24))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    conv3 = tf.nn.relu(conv3)
    #conv3 = tf.nn.dropout(conv3, keep_prob)
    #19,19,24 -- 15,15,30
    conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 30), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(30))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
    
    conv4 = tf.nn.relu(conv4)
    #conv4 = tf.nn.dropout(conv4, keep_prob)
    #15,15,30 --- 10,10,40
    conv5_W = tf.Variable(tf.truncated_normal(shape=(6, 6, 30, 40), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(40))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
    
    conv5 = tf.nn.relu(conv5)
    #conv5 = tf.nn.dropout(conv5, keep_prob)

    ##10x10x40 = 4000
    fc0   = flatten(conv5)
    
    ##4000 to 1000
    fc1_W = tf.Variable(tf.truncated_normal(shape=(4000, 1000), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1000))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1, keep_prob)

    ## 1000 -- 500
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1000,500), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(500))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    ## 500 -- 100
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(500,100), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(100))
    fc3    = tf.matmul(fc2, fc3_W) + fc3_b
    
    fc3    = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob)
    
    #100 -- 3
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(100, 4), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(4))
    logits = tf.matmul(fc3, fc4_W) + fc4_b
    
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32,1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 4)
keep_prob = tf.placeholder(tf.float32)


rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
top1 = tf.nn.top_k(logits, k=1)



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1 })
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_nor_gray)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        x_nor_gray, y_train = shuffle(x_nor_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_nor_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        training_accuracy = evaluate(x_nor_gray, y_train)    
        #validation_accuracy = evaluate(xv_nor_gray, y_valid)
        print("EPOCH {} ...".format(i+1))
        #print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("training Accuracy = {:.3f}".format(training_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


def prediction(x_prec):
    sess.run(correct_prediction, feed_dict={x:x_prec})

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    ll = x_nor_gray[2].reshape((1,32,32,1))
    test_prediction = prediction(ll)
    print(test_prediction)
    
ll = np.array(x_nor_gray[12]).reshape(1,32,32,1)

ll = x_nor_gray[0:12]



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    prec = sess.run(tf.argmax(logits, 1), feed_dict={x:ll,keep_prob: 1})




