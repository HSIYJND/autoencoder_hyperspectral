#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:55:04 2018

@author: abhi
"""


######Approach 1 ################################## (Error in identifying image)
from PIL import Image
im = Image.open('/home/abhi/Desktop/Anand/SUBSET_RF_1.tif')
im.show()

######Approach 2 ################################## (Error in identifying image)
import matplotlib.pyplot as plt
I = plt.imread('/home/abhi/Desktop/Anand/SUBSET_RF_1.tif')

######Approach 3 ################################## (Reads only one layer)
from osgeo import gdal
dataset = gdal.Open('/home/abhi/Desktop/Anand/SUBSET_RF_1.tif', gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()
    


######Approach 4 ################################## 
import gi
from osgeo import gdal
import numpy as np
ds = gdal.Open('/home/abhi/Desktop/Anand/SUBSET_RF_1.tif')
for i in (1,200):
    channel[i] = np.array(ds.GetRasterBand(i).ReadAsArray())