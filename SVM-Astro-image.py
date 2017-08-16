#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
用SVM对实时天文图像进行分类
用法:	1.使用pro_noise产生带有噪声的图片数据
	2.对有噪声和无噪声的已有图片(带标签)进行pro_data,保存到某一txt
	3.在sVM下进行训练,找到最优的惩罚系数c,损失函数g
	4.利用pro_noise产生实时的图片,使用SVM-Astro-inage进行实时的筛选
**注/usr/bin/libsvm-3.21/tools/下的easy.py寻找最优g,c
use in python2.7 by Awen in 2017.8.16
"""
import os,io,types,math,glob,shutil  #调用函数库
from astropy.io import fits
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from skimage import morphology,measure
from pycuda.compiler import SourceModule
from svmutil import *
from svm import *

import time
 
#start = time()
#print("Start: " + str(start))

mod = SourceModule("""
__global__ void Fun_snr(float *S_td, float *f_picture,int *N,int *Modesize)
{
	const int Imgw = N[0,0];
	const int modesize = Modesize[0,0];
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float s_td=0.0;
	float sum=0.0;   
	if(blockIdx.x <4060&& !((i+1)%Imgw/(Imgw-modesize+1)))
		{
		for(int p=0 ; p<modesize ; p++)
			for(int q=0 ; q<modesize ; q++)
			{
			sum  =sum + f_picture[ (blockIdx.x+Imgw/blockDim.x*p) * blockDim.x + threadIdx.x + q];	
			}
		sum = sum /pow(float(modesize),2);
		for(int p=0 ; p<modesize ; p++)
			for(int q=0 ; q<modesize  ; q++)
			{
			s_td =s_td + pow((f_picture[ (blockIdx.x+Imgw/blockDim.x*p)  * blockDim.x + threadIdx.x + q]-sum),2);
	 		}
		S_td[i]= sqrt(s_td / pow(float(modesize),2));
		}
	else
		{
			S_td[i]=0;
		}
}
__global__ void Fun_ROAD(float *ROAD, float *f_picture,int *N)
{	
	const int Imgw = N[0,0];
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockIdx.x <4088&& !((i+1)%Imgw/(Imgw-2)))
		{
		float a[9];
		int a_i=1;
		for(int p=0 ; p<3 ; p++)
			for(int q=0 ; q<3; q++)
			{
			a[a_i]=f_picture[(blockIdx.x+Imgw/blockDim.x*p) * blockDim.x + threadIdx.x + q]-f_picture[(blockIdx.x+Imgw/blockDim.x*1) * blockDim.x + threadIdx.x + 1];
			a[a_i]=abs(a[a_i]);
			a_i+=1;
	 		}
	 		{
				int ii,j;
				float t;
				for(ii=0;ii<8;ii++)
					for(j=ii+1;j<9;j++)
					{
						if(a[ii]>a[j])
						{
						t=a[ii];
						a[ii]=a[j];
						a[j]=t;
						}
					}
			}
		float a_sum=0;
		for(int kk=0;kk<5;kk++)
			a_sum=a_sum+a[kk];
		ROAD[i]= a_sum;
		}
	else
		{
			ROAD[i]=0;
		}
		__syncthreads();
}
""")
Fun_snr= mod.get_function("Fun_snr")
Fun_ROAD= mod.get_function("Fun_ROAD")
#纹理检测
def WL0(f):
	w=len(f)
	l=len(f[0])
	fmin=np.min(f)
	ones=f*0+65537-65536 #将数据类型由f的uint16转化为uint32
	g=f-ones*np.min(f)
	f1=g*(w-1)/np.max(g)
	f1=f1.astype(np.int16)
	#print f1,np.max(f1),np.min(f1)
	t=f*0
	#print f1,type(f1)
	for i in range(w):
		for j in range(l-1):
			t[f1[i,j+1],f1[i,j]] +=1
			t[f1[i,j],f1[i,j+1]] +=1
	#print t
	ASM=0.0
	CON=0.0
	IDF=0.0
	Hxy=0.0
	s=sum(sum(t))+0.0
	#print "S=",s
	for x in range(w):
		for y in range(l):
			ASM=ASM+pow(t[x,y],2)/pow(s,2)
			IDF=IDF+t[x,y]/s/(1+pow((x-y),2))
			if t[x,y]!=0:
				Hxy=Hxy-t[x,y]/s*(np.log(t[x,y]/s)/np.log(2))
			if 0<=y+x & y+x<=w-1 :
				CON=CON+pow(x,2)*t[y,y+x]
			if 0<=y-x & y-x<=w-1:
				CON=CON+pow(x,2)*t[y,y-x]
	ASM=ASM*10000	
	CON=np.log(CON)
	IDF=IDF*100
	#print "ASM=","%0.3f"%ASM,"CON=","%0.1f"%CON,"IDF=",IDF,"Hxy=","%0.1f"%Hxy
	return float("%0.1f"%ASM),float("%0.1f"%CON),float("%0.1f"%IDF),float("%0.1f"%Hxy)
#线状缺陷检测
def Lt(f):
	zz=np.mean(f)+2*np.std(f)
	f[f<zz]=0
	f[f>=zz]=1
	#实施骨架算法
	skeleton =morphology.skeletonize(f)
	dst11=morphology.remove_small_objects(skeleton,min_size=15,connectivity=8)
	labels=measure.label(dst11,8)  #8连通区域标记
	#print '第',k,'副图存在的条状缺陷个数为:',labels.max() #显示连通区域块数(从0开始标记)
	#for i in range(0,labels.max()):
	#	print '    第',i+1,'条长度为',len(labels[labels==i+1])
	return float("%0.1f"%labels.max())

#数据传入,参数设定
rec = []
#输入图像的路径
##########
dl=0
ul=1000
##########
d=str(dl)
u=str(ul)
path = r'/home/awen/桌面/CCD_modpicture/001test/*.fit*'
#产生的数据保存
#str1="/home/awen/桌面/CCD_modpicture/test001.txt"
#file1=open(str1,'w+')  

y, x = svm_read_problem('/home/awen/桌面/CCD_modpicture/train.txt')
prob  = svm_problem(y, x)
#c与g的值有ezsy.py对train训练所得
param = svm_parameter('-t 0 -c 2 -g 0.0078125 -q') #线性核 cost参数(1)   是否估算正确概率(0)
model = svm_train(prob,param)   #训练好的SVM模型
ww=0
while ww<3:
#按读入顺序排列
	if __name__== '__main__':
	    rec = glob.glob(path)
	    rec.sort() 
	length=len(rec)
	if length>0:
		print "该",path,"下现在有",length,"副图片"
		for k in range(0,length):
			hh=fits.open(rec[k]) #read the data of fits
			print k,rec[k]
			f=hh[0].data
			f=f*256.0/np.max(f)
			Imgw=len(f) 
			Imgl=len(f[0])
			modesize=9  
			fw=f.astype(np.float)
			wenli0=WL0(fw)
			Linet=Lt(f)

			#把Imgw作为数组传入cuda
			N=np.zeros([1,1])
			N[0,0]=Imgw
			N= np.int32(N)  

			#把modesize传入cuda
			Modesize=np.zeros([1,1])
			Modesize[0,0]=modesize
			Modesize=np.int32(Modesize)  

			#将f一维化传入cuda
			f=f.astype(np.float32)
			f_picture=f.flatten()  #一维化
			S_td=np.zeros_like(f_picture).astype(np.float32)
			ROAD=np.zeros_like(f_picture).astype(np.float32)

			# GPU run
			nTheads = 256   
			nBlocks = Imgw*Imgl/nTheads
			Fun_snr(drv.Out(S_td),drv.In(f_picture),drv.In(N),drv.In(Modesize), block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ,1) )
			Fun_ROAD(drv.Out(ROAD), drv.In(f_picture),drv.In(N), block=(nTheads, 1, 1 ), grid=( nBlocks, 1 ,1) )
			ROAD.sort()
			ROAD=ROAD*0.1
			S_td=S_td[S_td>0]
			snrmy=10*math.log(max(S_td)/min(S_td))
			#wenli='0 1:'+str(float("%0.1f"%snrmy))+' 2:'+str(float("%0.1f"%np.max(ROAD)))+' 3:'+str(float("%0.1f"%Linet))+' 4:'+str("%0.1f"wenli0[0]*0.1)+' 5:'+str(wenli0[1])+' 6:'+str(wenli0[2])+' 7:'+str(wenli0[3]*10)
			yt=[1]
			xt =[{1: float("%0.1f"%snrmy), 2: float("%0.1f"%np.max(ROAD)), 3: float("%0.1f"%Linet), 4: float("%0.1f"%(wenli0[0]*0.1)), 5: wenli0[1], 6: wenli0[2], 7: wenli0[3]}]
			print xt
			p_label, p_acc, p_val = svm_predict(yt,xt, model)
			
			if p_acc[0]<1:
				shutil.move('/home/awen/桌面/CCD_modpicture/001test/'+os.path.basename(rec[k]),'/home/awen/桌面/CCD_modpicture/001bad/'+os.path.basename(rec[k]))
			else:
				shutil.move('/home/awen/桌面/CCD_modpicture/001test/'+os.path.basename(rec[k]),'/home/awen/桌面/CCD_modpicture/001god/'+os.path.basename(rec[k]))				
				#shutil.move('/home/awen/Awen/tdate/test2/test2.'+u+'/'+os.path.basename(rec[k]),'/home/awen/Awen/tdate/test2/test2.'+u+'god/'+os.path.basename(rec[k]))
				
			#file1.write("%s\n"%xt)
		ww=0
	else:
		time.sleep(10)
		ww+=1
stop = time()
print("Stop: " + str(stop))
print(str(stop-start) + "秒")
