"""
Created on Thu Sep 20 11:39:39 2018

@author: abhi

An example of 3d-convolutional autoencoder
The autoencoder network is pretrained by nearly 1.9 million unlabelled hyperion patches.
As for the classification of Indian Pines dataset, autoencoder network is trained 
again by Indian Pines dataset patches without label. Then the encoder part is extracted 
and used to produce encoded features. Finally, features generated from convolutional 
encoder is fed to softmax classifier.
"""

import numpy as np
import os
import time
from sklearn import svm
from sklearn.metrics import cohen_kappa_score
from keras.callbacks import EarlyStopping
from sklearn import svm
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
import h5py as h5

num_bands = 200
num_classes = 16
NUM = 0.3

#all the paths
I_vec = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')
I_gt = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Indian_pines_gt.npy')
#Model saving
path_ip = r"/home/abhi/Documents/Hyper/Dataset_Hyperspectral/"
path_cae_save = os.path.join(path_ip, "cae_model_2d_12.h5")
path_encoder_save = os.path.join(path_ip, "encoder_model_2d_12.h5")
path_cae_lab_save = os.path.join(path_ip, "cae_model_lab_2d_12.h5")
path_encoder_lab_save = os.path.join(path_ip, "encoder_model_lab_2d_12.h5")


# making 8x8x200 patches out of the main image and its ground truth
'''
igt=np.ravel(I_gt)
I_vect=I_vec.transpose(2,0,1).reshape(200,-1).T



#Extra info getting from the numpy arrays
pines_sampleno=np.sort(igt)
pines_sort=np.sort(igt)
pines_indices=np.argsort(igt)
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

ix_train=np.concatenate((ip_1,ip_2,ip_3,ip_4,ip_5,ip_6,ip_7,ip_8,ip_9,ip_10,ip_11,ip_12,ip_13,ip_14,ip_15,ip_16))
iy_train=np.concatenate((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10,y_11,y_12,y_13,y_14,y_15,y_16))
ix_test=np.concatenate((ip_1_test,ip_2_test,ip_3_test,ip_4_test,ip_5_test,ip_6_test,ip_7_test,ip_8_test,ip_9_test,ip_10_test,ip_11_test,ip_12_test,ip_13_test,ip_14_test,ip_15_test,ip_16_test))
iy_test=np.concatenate((y_1_test,y_2_test,y_3_test,y_4_test,y_5_test,y_6_test,y_7_test,y_8_test,y_9_test,y_10_test,y_11_test,y_12_test,y_13_test,y_14_test,y_15_test,y_16_test))
'''


input_img = Input(shape=(num_bands , 8, 8))

x = Conv2D( 144, (3, 3),activation='relu', padding='same', data_format = "channels_first")(input_img)
x = Conv2D( 88, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D( 44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D( 22, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)
x = Conv2D( 12, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
encoded = MaxPooling2D((2, 2), padding='same', data_format = "channels_first")(x)

# at this point the representation is (12, 1, 1) i.e. 12-dimensional
x = UpSampling2D((2, 2), data_format = "channels_first")(encoded)
x = Conv2D(12, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D(22, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D(44, (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = UpSampling2D((2, 2), data_format = "channels_first")(x)
x = Conv2D( 88 , (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
x = Conv2D( 144 , (3, 3), activation='relu', padding='same', data_format = "channels_first")(x)
decoded = Conv2D(num_bands, (3, 3), activation='tanh', padding='same', data_format = "channels_first")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics = ['accuracy'])

print(autoencoder)
# Setting when to stop training
early_stopping = EarlyStopping(monitor='loss', patience=5)

# Training with unlabelled data
#array_patches = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')
array_patches = ii
cae = autoencoder.fit(x=array_patches, y=array_patches, batch_size=100, epochs=250, callbacks=[early_stopping])

# Save the trained model
#autoencoder.save(path_cae_save)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
encoder.save(path_encoder_save)


# Train again using labeled data
array_patches_lab = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Indian_pines_gt.npy')
array_patches = ii
cae = autoencoder.fit(x=array_patches_lab, y=array_patches_lab, batch_size=100, epochs=250, callbacks=[early_stopping])


# Save the trained model
#autoencoder.save(path_cae_lab_save)

encoder = h5.File("/home/abhi/Documents/Hyper/Autoencoder/encoder_model_lab_2d_12.h5", "r")
# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)
#encoder.save(path_encoder_lab_save)


# Produce encoded features
x_train_co = encoder.predict(I_vec)
x_test_co = encoder.predict(ix_test)          
x_train_co = x_train_co.reshape((x_train_co.shape[0], x_train_co.shape[1]))
x_test_co = x_test_co.reshape((x_test_co.shape[0], x_test_co.shape[1]))

'''
# Train the classifier
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#scores = [3.0, 1.0, 0.2]
print(softmax(ixtrain))

# Predict lables based on image data
y_predict=softmax.predict(x_test_co)
kappa_value = cohen_kappa_score(y_predict, y_test)

print(kappa_value)
'''
clf_svm = svm.SVC()
clf_svm.fit(x_train_co, y_train)

# Predict lables based on image data
y_predict=clf_svm.predict(x_test_co)
kappa_value = cohen_kappa_score(y_predict, y_test)

print(kappa_value)