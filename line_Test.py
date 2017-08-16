#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
检测某个文件中的图片线状缺陷个数
by Awen Python2.7 2017.8.16
"""
import cv2,glob
from skimage import morphology,draw,measure,color
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

path='/home/awen/桌面/CCD_modpicture/line2/*.fit*'
if __name__== '__main__':
    rec = glob.glob(path)
    rec.sort() 
length=len(rec)
print "该",path,"下有",length,"副图片"
for k in range(0,length):
	hh=fits.open(rec[k]) #read the data of fits
	f=hh[0].data
	f=hh[0].data
	zz=np.mean(f)+2*np.std(f)
	f[f<zz]=0
	f[f>=zz]=1

	#实施骨架算法
	skeleton =morphology.skeletonize(f)
	dst11=morphology.remove_small_objects(skeleton,min_size=15,connectivity=8)

	labels=measure.label(dst11,8)  #8连通区域标记
	print '第',k,'副图存在的条状缺陷个数为:',labels.max() #显示连通区域块数(从0开始标记)

	for i in range(0,labels.max()):
		print '    第',i+1,'条长度为',len(labels[labels==i+1])
