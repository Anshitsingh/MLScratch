
# coding: utf-8

# In[5]:



import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import PIL.Image
from IPython.display import display
import math
import os
import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from io import BytesIO


# In[6]:


def DataBase_creator(archivezip, nwigth, nheight, save_name):
    
    s = (len(archivezip.namelist()[:])-1, nwigth, nheight,3) 
    allImage = np.zeros(s)

    for i in range(1,len(archivezip.namelist()[:])):
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(filename) 
        image = image.resize((nwigth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) 

        allImage[i-1]=image

    pickle.dump(allImage, open( save_name + '.p', "wb" ) )
    




# In[7]:


y_train=pd.read_csv("/home/jash/Desktop/capstone/labels.csv")
y_train=pd.get_dummies(y_train["breed"]).values


# In[8]:


archive_train = ZipFile("/home/jash/Desktop/capstone/train.zip", 'r')
image_resize = 40
DataBase_creator(archivezip = archive_train, nwigth = image_resize, nheight = image_resize , save_name = "x_train")
x_train = pickle.load( open( "x_train.p", "rb" ) )
x_train.shape


# In[9]:


def batch(x,y,start,batch_size):
    x_batch=x[start:start+batch_size]
    y_batch=y[start:start+batch_size]
    start=start+batch_size
    return x_batch,y_batch,start

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# In[10]:


def weight_variable(shape):
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)

def bias_variable(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)



def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
 
    shape = [filter_size, filter_size, num_input_channels, num_filters]
 
    weights = weight_variable(shape)
    biases = bias_variable([num_filters])

    layer = tf.nn.relu(tf.nn.conv2d(input=input,
                                    filter=weights,
                                    strides=[1, 2, 2, 1],
                                    padding='SAME') + biases)

    if use_pooling: 
        return max_pool(layer), weights

        

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 
    weights = weight_variable([num_inputs, num_outputs])
    biases = bias_variable([num_outputs])
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
 
    return layer


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.2, random_state=4)


# In[12]:


x = tf.placeholder(tf.float32, shape=[None, image_resize,image_resize,3], name='input_data')
x_image = tf.reshape(x, [-1,image_resize,image_resize,3])
# correct labels
y_ = tf.placeholder(tf.float32, shape=[None, 120], name='correct_labels')

# fist conv layer
convlayer1, w1 = new_conv_layer(x_image, 3, 3, 30)
# second conv layer
convlayer2, w2 = new_conv_layer(convlayer1, 30, 3, 50)
#third conv layer

convlayer3,w3=new_conv_layer(convlayer2,50,3,30)

# flat layer
flat_layer, num_features = flatten_layer(convlayer2)
# fully connected layer
fclayer = new_fc_layer(flat_layer, num_features, 1024)


# DROPOUT
keep_prob = tf.placeholder(tf.float32)
drop_layer = tf.nn.dropout(fclayer, keep_prob)
# final layer
W_f = weight_variable([1024, 120])
b_f = bias_variable([120])
y_f = tf.matmul(drop_layer, W_f) + b_f
y_f_softmax = tf.nn.softmax(y_f)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_f))

# train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_f_softmax, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init
init = tf.global_variables_initializer()



# In[14]:


num_steps =40
batch_size =200
test_size =10
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)
    
    for epoch in range(50):
        start=0
        for step in range(num_steps):
            x_batch,y_batch,start= batch(x_train,y_train,start,batch_size)
         
        
            if step % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                x:x_batch, y_: y_batch, keep_prob: 1.0})
            
           
            train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
        print('Epoch %d, training accuracy %f' %(epoch, train_accuracy))
   
    test_accuracy = 0.0
    start=0
    for i in xrange(test_size):
        x_batch,y_batch,start= batch(x_test,y_test,start,batch_size)
        acc = accuracy.eval(feed_dict={x: x_batch, y_:y_batch, keep_prob: 1.0})
        if i % 10 == 0:
            print('%d: test accuracy %f' % (i, acc))
        test_accuracy += acc
    print 'avg test accuracy:', test_accuracy/(test_size)
    file_writer = tf.summary.FileWriter('/home/jash/Desktop/capstone', sess.graph)
    file_writer.add_graph(sess.graph)


