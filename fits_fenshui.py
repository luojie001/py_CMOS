#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
检测单副图片中的现状缺陷个数
by Awen Python2.7 2017.8.16
"""
import cv2
from skimage import morphology,draw,measure,color
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
hh=fits.open('/home/awen/Mosaick/B类条状缺陷/f.fits')
#hh=fits.open('/home/awen/桌面/CCD_modpicture/line2/line75.fits')
f=hh[0].data
zz=np.mean(f)+2*np.std(f)
f[f<zz]=0
f[f>=zz]=1

#实施骨架算法
skeleton =morphology.skeletonize(f)
dst11=morphology.remove_small_objects(skeleton,min_size=15,connectivity=8)

labels=measure.label(dst11,8)  #8连通区域标记
print '该图片中存在的条状缺陷个数为:',labels.max() #显示连通区域块数(从0开始标记)

for i in range(0,labels.max()):
	print '    第',i+1,'条长度为',len(labels[labels==i+1])

#显示
#plt.imshow(dst11,plt.cm.gray)
#plt.show()


"""
#计算中轴和距离变换值
skel, distance =morphology.medial_axis(skeleton, return_distance=True)

#中轴上的点到背景像素点的距离
dist_on_skel = distance * skel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
#用光谱色显示中轴
ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
ax2.contour(skeleton, [0.5], colors='g')  #显示轮廓线

fig.tight_layout()
plt.show()
"""
