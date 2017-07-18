#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
2017.6.22 备份 两种泽尼克相差+赛德尔相差
"""


"""This is a CCD module simulation code used for wide field telescope simulation
********************************************************************************
Maintained by Peng Jia
Email: robinmartin20@gmail.com
********************************************************************************
Assuming all the fits files in this folder will be transformed into CCD images
**Method: Images are pdf of photons. 
	  According to the source information and Detector information, images
will be transfered to CCD images.
********************************************************************************
"""
"Necessary packages"
import numpy as np
import os,io,types,math,PhysicalOpticalProp,logging
from astropy.io import fits
from scipy.special import gamma 
from scipy import interpolate
from scipy import ndimage
import scipy.interpolate as interp
import scipy.ndimage.interpolation as sn 
import scipy.stats as stats
from math import factorial

'***************************************************************************'
"Definition of Source"
class Source(object):
	def __init__(self,Magnitude,FilterBand,SkyBackground,ExposureTime):
		self.Magnitude=Magnitude
		self.FilterBand=FilterBand
		self.SkyBackground=SkyBackground
		self.ExposureTime=ExposureTime
		"""Set the FilterTable for reference
		      Photometric system
		       Band ( wavelength , bandwidth , zero point )
			 U  ( 0.360e-6 , 0.070e-6 , 2.0e12 )
			 B  ( 0.440e-6 , 0.100e-6 , 5.4e12 )
			 V0  ( 0.500e-6 , 0.090e-6 , 3.3e12 )
			 V  ( 0.550e-6 , 0.090e-6 , 3.3e12 )
			 R  ( 0.640e-6 , 0.150e-6 , 4.0e12 )
			 I  ( 0.790e-6 , 0.150e-6 , 2.7e12 )
			 J  ( 1.215e-6 , 0.260e-6 , 1.9e12 )
			 H  ( 1.654e-6 , 0.290e-6 , 1.1e12 )
			 Ks ( 2.157e-6 , 0.320e-6 , 5.5e11 )
			 K  ( 2.179e-6 , 0.410e-6 , 7.0e11 )
			 L  ( 3.547e-6 , 0.570e-6 , 2.5e11 )
			 M  ( 4.769e-6 , 0.450e-6 , 8.4e10 )
			 Na ( 0.589e-6 , 0        , 3.3e12 )
		"""
		self.FilterTable=dict([('U',(0.36e-6,0.070e-6,2.0e12)),('B',(0.44e-6,0.10e-6,5.4e12)),('V0',(0.50e-6,0.090e-6,3.3e12)),('V',(0.55e-6,0.090e-6,3.3e12)),('R',(0.64e-6,0.15e-6,4.0e12)),('I',(0.79e-6,0.15e-6,2.7e12)),('J',(1.215e-6,0.26e-6,1.9e12)),('H',(1.654e-6,0.29e-6,1.1e12)),('Ks',(2.157e-6,0.320e-6,5.5e11)),('K',(2.179e-6,0.41e-6,7.0e11)),('L',(3.547e-6,0.570e-6,2.5e11)),('M',(4.769e-6,0.450e-6,8.4e10)),('Na',(0.589e-6,0,3.3e12))])
  #print "All components loaded correctly"
 #Function used for calculating all the photons from source
	def SourcePhotonApNornumber(self):
		if self.FilterBand in self.FilterTable:
			print('Found Filter Name with {}\n'.format(self.FilterBand))
   			print('The central Wavelength is {}\n'.format(self.FilterTable[self.FilterBand][0]))
			print('The Bandwidth is {}\n'.format(self.FilterTable[self.FilterBand][1]))
			print('The base Photon number is {:.2E}\n'.format(Decimal(self.FilterTable[self.FilterBand][2])))
   #print(self.FilterTable[self.FilterBand][1])
		else:
			print "Illegal Input Filter Value"
		if self.ExposureTime<0:
			print('Exposure time is negative, please check that \n')
		else: 
			   print('Exposure time is {} \n'.format(self.ExposureTime))
			   SourceApNorPhoton=self.FilterTable[self.FilterBand][2]*10**(-0.4*self.Magnitude)*self.ExposureTime
			   print('Normalized Aperture Photon number is {:.2E} \n'.format(Decimal(SourceApNorPhoton)))
			   return SourceApNorPhoton
 
 #Function used for calculating all the photons from sky
	def SkyBackPhotonApNornumber(self):
		if self.FilterBand in self.FilterTable:
			print('Found Filter Name with {}\n'.format(self.FilterBand))
			print('The central Wavelength is {}\n'.format(self.FilterTable[self.FilterBand][0]))
			print('The Bandwidth is {}\n'.format(self.FilterTable[self.FilterBand][1]))
			print('The base Photon number is {:.2E}\n'.format(Decimal(self.FilterTable[self.FilterBand][2])))
			#print(self.FilterTable[self.FilterBand][1])
		else:
			print "Illegal Input Filter Value"
		if self.ExposureTime<0:
			print('Exposure time is negative, please check that \n')
		else:
			SkyApNorPhoton=self.FilterTable[self.FilterBand][2]*10**(-0.4*self.SkyBackground)*self.ExposureTime
			print('Normalized Aperture Photon number of sky is {:.2E} \n'.format(Decimal(SkyApNorPhoton)))
			return SkyApNorPhoton

'**********************************************************************************************'

def genzernike(u,nn):
	dx=2.0/(nn-1)
	x=np.linspace(-1,1,nn)
	y=np.linspace(-1,1,nn)
	xx,yy = np.meshgrid(x,x)
	theta=np.arctan2(yy,xx)
	r = np.hypot(xx,yy)
	rx=np.hypot(xx,yy)
	rx[rx>1]=0
	nn=int(nn)
	z=[[[0 for i in range(int(nn))] for i in range(int(nn))]for i in range(int((u+1)*(u+2)*0.5))]
	for n in range(0,u+1):
		for m in range(0,u+1):
			R=np.zeros([nn,nn])
			if (m==0)&(n==0):			
				J=0
				c0=np.sqrt((n+1)/np.pi)
				GaC=1
				rz=r
				rz[rz>1]=0
				rz[rz>0]=c0
				z[J]=rz
			elif (m==0)&(n!=0)&(np.mod(n,2)==0):
				J=int(math.ceil((n+1)*n*0.5))
				c0=np.sqrt((n+1)/np.pi)
				for s in range(0,int(math.ceil(n*0.5))):
					GaC=(-1)**s*gamma(n-s+1)/(gamma(s+1)*gamma((m+n)/2.0-s+1)*gamma((n-m)/2.0-s+1))			
					R=R+GaC*rx**(n-2*s)
				z[J]=c0*R
			if (m!=0)&(n>=m)&(np.mod((n-m),2)==0):
				J=(n+1)*n/2+m-1
				cm=np.sqrt(2*(n+1)/np.pi)
				temp=(n-m)*0.5
				for s in range(0,int(temp)+1):
					GaC=(-1)**s*gamma(n-s+1)/(gamma(s+1)*gamma((n+m)/2.0-s+1)*gamma((n-m)*0.5-s+1))				
					R=R+GaC*rx**(n-2*s)
				if (np.mod(J+1,2)==0):
					z[J]=cm*R*np.cos(m*theta)
					z[J+1]=cm*R*np.sin(m*theta)
				else:
					z[J]=cm*R*np.sin(m*theta)
					z[J+1]=cm*R*np.cos(m*theta)
	result=z[m-1]
	return result 

#####znk-github############
_log = logging.getLogger(__name__)
def R(n, m, rho):
    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if (n - m)&1:
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k)\
			 * factorial((n - m) / 2. - k)))
            output += coef * rho ** (n - 2 * k)
        return output

def zernike(n, m, npix, rho=None, theta=None, outside=0,
            noll_normalize=True):

    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        _log.warn("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    _log.debug("Zernike(n=%d, m=%d)" % (n, m))

    if theta is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    else:
        if rho is None:
            raise ValueError("If you provide a theta input array, you must also provide an array "
                             "r with the corresponding radii for each point.")

    aperture = np.ones(rho.shape)
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask
    if m == 0:
        if n == 0:
            genzernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            genzernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        genzernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        genzernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    genzernike_result[np.where(rho > 1)] = outside
    return genzernike_result
####-*---------------------------------------*-**
def seidel(M):
	#M=1024
	L=1e-3   #
	du=L/M   
	u=np.linspace(-L/2,L/2,L/du+1) #门于柱
	v=u
	lambda1=0.55e-6  #波长
	k=2*np.pi/lambda1
	Dxp=20e-3 #望远镜口径
	wxp=Dxp/2
	zxp=100e-3  #传输距离
	fnum=zxp/(2*wxp)
	lz=lambda1*zxp 
	twoof0=1/(lambda1*fnum)
	u0=.2
	v0=.3
	wd=0*lambda1
	w040=0.4*lambda1
	w131=7.637*lambda1
	w222=0.4*lambda1
	w220=7.536*lambda1
	w311=8.157*lambda1
	fu=np.linspace(-1/(2*du),1/(2*du)-1/L,(1/du-1/L)*L+1)
	[Fu,Fv]=np.meshgrid(fu,fu)
	#seidel_5
	beta=math.atan(v0/u0)
	u0r=np.sqrt(u0**2+v0**2)
	X=-lz*Fu/wxp
	Y=-lz*Fv/wxp
	Xr=X*np.cos(beta)+Y*np.sin(beta)
	Yr=-X*np.sin(beta)+Y*np.cos(beta)
	rho2=Xr**2+Yr**2
	w=wd*rho2+w040*rho2**2+w131*u0r*rho2*Xr+w222*u0r**2*Xr**2+w222*u0r**2*rho2+w220*u0r**2*rho2+w311*u0r**3*Xr
	return w

def imresample(oldpixsize,img,CCDSize):
	r=len(img)
	c=len(img[0])
	# smaller variable names
	nimg=sn.zoom(img,CCDSize/len(img))
	#zfun_smooth_rbf = interp.Rbf(Ox, Oy,img, function='cubic', smooth=0)
	#nimg = zfun_smooth_rbf(Nx,Ny)
	return nimg
'Definition of Optical system'
class OpticalSystem(object):
	def __init__(self,Aperture,SecondaryMirror,Reflectivity,Aberration,CCDSample,SourceImg): #What parameters to contain in this part ??
		self.Aperture=Aperture
		self.SecondaryMirror=SecondaryMirror
		self.Reflectivity=Reflectivity
		self.Aberration=Aberration
		self.CCDSample=CCDSample
		self.SourceImg=SourceImg
	def TelescopeSimulation(self):
		#I参数传入,依次为望远镜口径\副镜\反射率\静态相差\采样率 	

		#II 产生望远镜的相差矩阵	
		NZer=len(self.Aberration)
		#模拟相差矩阵的大小
		CCDSize=self.CCDSample*self.Aperture
		print CCDSize
		"""
		#泽尼克相差
		Phase=np.zeros([int(CCDSize),int(CCDSize)])
		#循环残生相差矩阵
		for ind in range(0,NZer):
			#两种泽尼克
			Phase=Phase+Aberration[ind]*zernike(ind,ind,CCDSize)
			#Phase=Phase+Aberration[ind]*genzernike(ind,CCDSize)
		"""
		#seidel相差
		Phase=seidel(int(CCDSize))
		#III 根据相差产生PSF与最终图像
		#产生PSF
		zz=np.zeros([len(Phase),len(Phase[0])])
		Iphase=np.complex_(zz)
		Iphase.imag=Phase
		TelPSF=abs(np.fft.fftshift(np.fft.fft2(np.exp(Iphase))))
		TelPSF=TelPSF/np.sum(TelPSF)
		#图像重采样
		Imgw=len(self.SourceImg)
		Imgl=len(self.SourceImg[0])
		oldpixsize=[1.0,1.0]
		newpixsize=[Imgw*1.0/CCDSize,Imgl*1.0/CCDSize]
		NewImg=imresample(oldpixsize,self.SourceImg,CCDSize)
		#最终图像产生
		OutputImage=abs(ndimage.convolve(NewImg,TelPSF,mode='constant'))
		TransferRate=(1-(self.SecondaryMirror*1.0/self.Aperture)**2)*self.Reflectivity
		return OutputImage,TransferRate
	#ccd set
"""
	def AstroImgNoise1(self,OutputImage):
		#源图像参数传入
		Mag=Source[0]
		Zeropoint=Source[1]
		Img=Source[2]
		SkyB=Source[3]
		TeleA=Source[4]
		#CCD参数传入
		Type=CCD[0]
		RONs=CCD[1]
		CTE=CCD[2]
		DC=CCD[3]
		Gain=CCD[4]
		Bining=CCD[5]
		FWell=CCD[6]
		QE=CCD[7]
		Imgw=len(Img)
		Imgl=len(Img[0])
		if Type==1:
			Gain=Gain*np.ones([Imgw,Imgl])
		elif Type==2:
			RandAmp=0.01
			Gain =Gain*(np.ones([Imgw,Imgl])+RandAmp*np.random.random([Imgw,Imgl]))
		else:
			print "Wrong Type of CCD"
		#III 光度学计算 图像有效灰度值转化为ADU
		Nphoton=Zeropoint*10**(-0.4*Mag)*Exposuretime*np.pi*TeleA**2
		#背景光子数计算
		Bphoton=Zeropoint*10**(-0.4*SkyB)*Exposuretime*np.pi*TeleA**2
		#产生像素分布的概率密度函数pdf
		Imgpdf=np.round(Img*Nphoton/(np.sum(np.sum(Img))))
		#产生入射图像光电子矩阵
		Img=Imgpdf*QE+stats.poisson.rvs(mu=Bphoton/Imgw/Imgl,size=([Imgw,Imgl]))
		#IV 产生读出噪声矩阵RON (包含Bias和电路噪声,Possion分布)
		RONMatrix=stats.poisson.rvs(mu=RONs,size=([Imgw,Imgl]));
		#产生暗电流噪声矩阵 
		DCMatrix=date=stats.poisson.rvs(mu=DC*Exposuretime,size=([Imgw,Imgl]))
		#根据电荷转移率产生电荷损失矩阵
		CTI=(Imgw+Imgl)*(1-CTE)*(1+0.01*np.random.random([Imgw,Imgl]))*Img
		#根据增益和电荷矩阵产生读出电子图像矩阵
		Img=(Img-CTI)*Gain+RONMatrix+DCMatrix
		#根据CCD井深产生最终图像
		Img[Img>FWell]=FWell
		#IIV 根据是否bining确定输出矩阵大小
		Outimg=np.zeros([Imgw/Bining,Imgl/Bining])
		for i in range(0,Imgw/Bining):
			for j in range(0,Imgl/Bining):
				for k in range(0,Bining):
			    		Outimg[i,j]=Outimg[i,j]+Img[(i-1)*Bining+k,(j-1)*Bining+k]
	
		SignalNoiseRatio=20*np.log(Nphoton/(Imgw*Imgl*(RONs**2+DC)+Bphoton+Nphoton)**(0.5))
		#IIIV 最终输出图像
		return Outimg,SignalNoiseRatio
"""

######main######
import matplotlib.pyplot as plt
'''
产生不同信噪比的图像
模拟光学系统
测试图像的产生
'''
import numpy as np
TestImg=np.zeros([255,255])
TestImg[128,128]=1
hh=fits.open('/home/awen/桌面/psf/-0.20psf-5.00')
a=hh[0].data
TestImg=a
#基本参数设定
#*-*望远镜类*-*
Aperture=2
SecondaryMirror=0.4
Reflectivity=0.97
Aberration=[0,0,0,10,0,0,0,0,0,0]
CCDSample=4.5
Telescope=[Aperture,SecondaryMirror,Reflectivity,Aberration,CCDSample]

#source
Magnitude =10
Zeropoint = 1.1e12
#Img=OutputImg
SkyBackground=10
TeleA = 2
#Source=[Magnitude,Zeropoint,Img,SkyBackground,TeleA]

#CCD set
Type=1
RONs=5
CTE=0.99999
DC=10
Gain=1
Bining=1
FWell=60000
QE=0.97
CCD=[Type,RONs,CTE,DC,Gain,Bining,Bining,FWell,QE]
ExposureTime=0.00001

if __name__ == '__main__':
	ss=OpticalSystem(Aperture,SecondaryMirror,Reflectivity,Aberration,CCDSample,TestImg)
	OutputImg,TransferRate=ss.TelescopeSimulation()
	#OutImg,SNRImg=ss.AstroImgNoise(OutImg)
	print TransferRate
	print OutputImg
	import matplotlib.pyplot as plt
	plt.imshow(OutputImg)
	plt.show()








