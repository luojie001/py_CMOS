#!/usr/bin/python
# -*- coding: utf-8 -*-
#产生带5种带噪声的图片数据
import os,random,glob,math
from astropy.io import fits
import numpy as np
import scipy.stats as stats
from time import time

start = time()
##############cosmic_rays######################
def cosmic_rays(f,n):
	fmax=np.max(f)
	for n in range(n):
		dx=random.randint(0,len(f)-1)
		dy=random.randint(0,len(f[0])-1)
		f[dx,dy]=fmax
	return f
#################linear#######################
def linear(f,n):
	d=1
	fmax=np.max(f)
	for n in range(n):
		x=random.randint(0,len(f)-1)
		y=random.randint(0,len(f[0])-1)
		f[x:x+d,y:y+d]=fmax
		for j in range(50):
			dx=random.randint(0,1)
			dy=random.randint(-1,1)
			x=x+dx
			y=y+dy
			f[x:x+d,y:y+d]=fmax
	return f
########################  adcloud ###################
def randn(M,N):
	c_real=np.random.normal(0,1,(M,N))
	c_imag=np.random.normal(0,1,(M,N))
	cc=np.complex_(c_real)
	cc.imag=c_imag
	return cc
def ft_phase_screen(r0,N,delta,L0,l0):
	del_f = 1.0/(N*delta)
	fx = np.array(range(-N/2,N/2)) * del_f
	fx,fy = np.meshgrid(fx,fx)
	#th=np.arctan2(fy,fx)
	f = np.hypot(fx,fy)
	#fm = 5.92/l0/(2*math.pi)
	#f0 = 1.0/L0
	PSD_phi = 0.023*(r0**(-5.0/3))*(f**(-11.0/3))
	PSD_phi[N/2,N/2] = 0
	cn =math.sqrt(2)*randn(N,N)*np.sqrt(PSD_phi)*del_f
	phz = ift2(cn,1)
	return phz
def ift2(G,delta_l):
	Gpd=np.fft.ifftshift(G)
	Goverpad=np.fft.ifft2(Gpd)
	g=np.fft.ifftshift(Goverpad)*(N*delta_l)**2
	return g
def adcloud(f,n):
	phz_hi = ft_phase_screen(r0, N, delta, L0, l0)
	phz=phz_hi.real
	g=f*0
	for i in range(1,N):
		for j in range(1,N):
			if phz[i][j]>-n:
				g[i,j]=20*phz[i,j]
			else:
				g[i,j]=f[i,j]
	return g
D = 1.0
r0 = 0.1
N = 1024
L0 = 100
l0 = 0.01
delta = D/N
#################################AstroImgNoise##########################
if __name__ == '__main__':
	hh=fits.open('/home/awen/桌面/CCD_modpicture/CMOS_Signal_CCD_DN_py.fits')
	f=hh[0].data
	#f=cosmic_rays(f,5)
	#f=linear(f,3)
	f=adcloud(f,0)#[-10,10]
	show = fits.PrimaryHDU(f)
	showlist = fits.HDUList([show])
	showlist.writeto('/home/awen/桌面/CCD_modpicture/yun3.fits')
stop = time()
print(str(stop-start) + "秒")
