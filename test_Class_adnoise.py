#!/usr/bin/python
# -*- coding: utf-8 -*-

import random,math
from astropy.io import fits
import numpy as np
import scipy.stats as stats
from time import time
start = time()
class Adnoise:
	def __init__(self):
		self.D=1.0
		self.r0=0.1
		self.N=1024
		self.L0=100
		self.l0=0.01	
		self.delta= self.D/self.N
	def adcloud(self,f,n):
		N=len(f)
		M=len(f[0])
		del_f = 1.0/(N*self.delta)
		fx = np.array(range(-N/2,M/2))*del_f
		fx,fy = np.meshgrid(fx,fx)
		fzz = np.hypot(fx,fy)
		PSD_phi = 0.023*(self.r0**(-5.0/3))*(fzz**(-11.0/3))
		PSD_phi[N/2,N/2]=0
		c_real=np.random.normal(0,1,(N,M))
		c_imag=np.random.normal(0,1,(N,M))
		cc=np.complex_(c_real)
		cc.imag=c_imag
		cn =math.sqrt(2)*cc*np.sqrt(PSD_phi)*del_f
		phz_hi=np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(cn)))*(len(cn)*1)**2
		phz=phz_hi.real
		g=f*0
		for i in range(1,N):
			for j in range(1,M):
				if phz[i][j]>-n:
					g[i,j]=20*phz[i,j]
				else:
					g[i,j]=f[i,j]
		return g
	def adcosmic_rays(self,f,n):
		for n in range(n):
			dx=random.randint(0,len(f)-1)
			dy=random.randint(0,len(f[0])-1)
			print n,dx,dy
			f[dx,dy]=np.mean(f)+2*np.std(f)+np.random.uniform(10,30)

		return f
	def adline(self,f,n):
		for n in range(n):
			x1=random.randint(100,len(f)-100)
			y1=random.randint(100,len(f[0])-100)
			x2=x1+random.randint(-50,50)
			y2=y1+random.randint(-50,50)
			res=MyBresenhamGetP(f,x1,y1,x2,y2)
			for i in range(len(res)):
				f[res[i][0],res[i][1]]=np.mean(f)+2*np.std(f)+np.random.uniform(10,30)
		return f

def  MyBresenhamGetP(img,x1,y1,x2,y2):
	"""
	直线取点算法,img为输入图像,P1(x1,y1),P2(x2,y2)为直线的两端点
	"""
	res=[]
	res.append([x1,y1])
	k = 1.0*(y1-y2)/(x1-x2)
	flag = 1
	t=0
	if (np.abs(k)>1):
		t=x1
		x1=y1
		y1=t
		t=x2
		x2=y2
		y2=t
	 	k = 1/k
	  	flag =0
	if (x1>x2):
		t=x1
		x1=y1
		y1=t
		t=x2
		x2=y2
		y2=t
	deltaX=x2-x1
	deltaY=np.abs(y2-y1)
	p=2*deltaY-deltaX
	minX=min(x1,x2)
	maxX=max(x1,x2)
	yk=y1
	for ii in range(minX,maxX):
		if p<0:
			p=p+2*deltaY
		else:
			if k>0:
				yk=yk+1
			else:
				yk=yk-1
			p=p+2*deltaY-2*deltaX
		if flag:
			res.append([ii+1,yk])
		else:
			res.append([yk,ii+1])
	return res

#################################Adnoise##########################
if __name__ == '__main__':
	hh=fits.open('/home/awen/桌面/CCD_modpicture/CMOS_Signal_CCD_DN_py.fits')
	f=hh[0].data
	aa=Adnoise()
	f=aa.adcloud(f, random.uniform(-10,10)) #在[-10,10]取一个随机数作为阈值
	f=aa.adcosmic_rays(f,5)
	f=aa.adline(f,5)
	show = fits.PrimaryHDU(f)
	showlist = fits.HDUList([show])
	showlist.writeto('/home/awen/桌面/CCD_modpicture/adnoise_test.fits')
stop = time()
print(str(stop-start) + "秒")
