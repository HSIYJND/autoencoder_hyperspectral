#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:44:11 2018

@author: abhi
"""

from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

NUM=0.5

I_vec = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')
I_gt = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Indian_pines_gt.npy')

igt=np.ravel(I_gt)
I_vect=I_vec.transpose(2,0,1).reshape(200,-1).T
#To verify the number of samples present in dataset and the unclassified points
pines_sampleno=np.sort(igt)
pines_sort=np.sort(igt)

#To get all the indices where a specific class is present
pines_indices=np.argsort(igt)


#Array slicing to get each class of Indian Pines
pine_number=pines_sampleno[10776:]
pine_ind=pines_indices[10776:]



a=pine_ind[0:46]
Class_1= I_vect[a]

b=pine_ind[46:1474]
Class_2= I_vect[b]

c=pine_ind[1474:2304]
Class_3= I_vect[c]

d=pine_ind[2304:2541]
Class_4= I_vect[d]

e=pine_ind[2541:3024]
Class_5= I_vect[e]

f=pine_ind[3024:3754]
Class_6= I_vect[f]

g=pine_ind[3754:3782]
Class_7= I_vect[g]

h=pine_ind[3782:4260]
Class_8= I_vect[h]

i=pine_ind[4260:4280]
Class_9= I_vect[i]

j=pine_ind[4280:5252]
Class_10= I_vect[j]

k=pine_ind[5252:7707]
Class_11= I_vect[k]

l=pine_ind[7707:8300]
Class_12= I_vect[l]

m=pine_ind[8300:8505]
Class_13= I_vect[m]

n=pine_ind[8505:9770]
Class_14= I_vect[n]

o=pine_ind[9770:10156]
Class_15= I_vect[o]

p=pine_ind[10156:10249]
Class_16= I_vect[p]

gt_1=igt[a]
gt_2=igt[b]
gt_3=igt[c]
gt_4=igt[d]
gt_5=igt[e]
gt_6=igt[f]
gt_7=igt[g]
gt_8=igt[h]
gt_9=igt[i]
gt_10=igt[j]
gt_11=igt[k]
gt_12=igt[l]
gt_13=igt[m]
gt_14=igt[n]
gt_15=igt[o]
gt_16=igt[p]
        

ip_1, ip_1_test, y_1, y_1_test = train_test_split(Class_1 , gt_1, test_size=NUM)
ip_2, ip_2_test, y_2, y_2_test = train_test_split(Class_2 ,gt_2, test_size=NUM)
ip_3, ip_3_test, y_3, y_3_test = train_test_split(Class_3 ,gt_3, test_size=NUM)
ip_4, ip_4_test, y_4, y_4_test = train_test_split(Class_4 ,gt_4, test_size=NUM)
ip_5, ip_5_test, y_5, y_5_test = train_test_split(Class_5 ,gt_5, test_size=NUM)
ip_6, ip_6_test, y_6, y_6_test = train_test_split(Class_6 ,gt_6, test_size=NUM)
ip_7, ip_7_test, y_7, y_7_test = train_test_split(Class_7 ,gt_7, test_size=NUM)
ip_8, ip_8_test, y_8, y_8_test = train_test_split(Class_8 ,gt_8, test_size=NUM)
ip_9, ip_9_test, y_9, y_9_test = train_test_split(Class_9 ,gt_9, test_size=NUM)
ip_10, ip_10_test, y_10, y_10_test = train_test_split(Class_10 ,gt_10, test_size=NUM)
ip_11, ip_11_test, y_11, y_11_test = train_test_split(Class_11 ,gt_11, test_size=NUM)
ip_12, ip_12_test, y_12, y_12_test = train_test_split(Class_12 ,gt_12, test_size=NUM)
ip_13, ip_13_test, y_13, y_13_test = train_test_split(Class_13 ,gt_13, test_size=NUM)
ip_14, ip_14_test, y_14, y_14_test = train_test_split(Class_14 ,gt_14, test_size=NUM)
ip_15, ip_15_test, y_15, y_15_test = train_test_split(Class_15 ,gt_15, test_size=NUM)
ip_16, ip_16_test, y_16, y_16_test = train_test_split(Class_16 ,gt_16, test_size=NUM)

x_train=np.concatenate((ip_1,ip_2,ip_3,ip_4,ip_5,ip_6,ip_7,ip_8,ip_9,ip_10,ip_11,ip_12,ip_13,ip_14,ip_15,ip_16))
y_train=np.concatenate((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10,y_11,y_12,y_13,y_14,y_15,y_16))
x_test=np.concatenate((ip_1_test,ip_2_test,ip_3_test,ip_4_test,ip_5_test,ip_6_test,ip_7_test,ip_8_test,ip_9_test,ip_10_test,ip_11_test,ip_12_test,ip_13_test,ip_14_test,ip_15_test,ip_16_test))
y_test=np.concatenate((y_1_test,y_2_test,y_3_test,y_4_test,y_5_test,y_6_test,y_7_test,y_8_test,y_9_test,y_10_test,y_11_test,y_12_test,y_13_test,y_14_test,y_15_test,y_16_test))


#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

'''
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''