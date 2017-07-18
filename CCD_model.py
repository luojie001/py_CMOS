#!/usr/bin/python
# -*-coding:utf-8 -*-

####<------------>###############
#CCD_model
"""photo-->charge-->voltage-->number"""
#tol_define_metrics()
import numpy as np
import signal
from numpy.matlib import repmat
from astropy.io import fits

#<-----define metric
m  = 1
cm = 1e-2*m
mm = 1e-3*m
mum = 1e-6*m
nm = 1e-9*m
rad = 1
mrad = 1e-3*rad
#----------------->
N=256
M=256

#illumination
input_screen_size = 1*m
input_screen_hole_size = 0.7*m
input_screen_blur = 0.2*m
amplitude_coeff = 0.1

#flag
photonshotnoise = 1
fPRNU  = 1
darkcurrent= 1
darkcurrent_Dshot  = 1
darkcurrent_DarkFPN_pixel = 1
darkcurrent_offsetFPN  = 1
sensenoderesetnoise  = 1
   #--plots
irradiance=1
electrons=1
volts=1 
fDN=1
plots=[irradiance,electrons,volts,fDN]
darkframe = 0
Venonlinearity = 0 #<--Gain non-linearity[CMOS ONLY!!!!]
VVnonlinearity= 0  #<--Gain non-linearity[CMOS ONLY!!!!]
ADCnonlinearity=0  #<--turn the non-linea
#noise

modle='Janesick-Gaussian'
nparameters=[]
factor=0.01
n_noisematrix=np.zeros([M,N])
nPRNU=[modle,nparameters,factor,n_noisematrix]

nDN=0.3
model = 'Janesick-Gaussian'
dparameters = []
noisematrix=0
darkFPN=[nDN,model,dparameters,noisematrix]
"""
model      = 'LogNormal';
parameters = [0, 0.4]; 
 
model       = 'Wald';
parameters  = 2;
"""
#darkFPN_offset
model  = 'Janesick-Gaussian'
parameters = []
DNcolumn       = 0.0005
dfnoisematrix=0
darkFPN_offset=[model,parameters,DNcolumn,dfnoisematrix]
Factor = 0.8
sigma_ktc=0
sr_noisematrix=0
sn_reset=[Factor,sigma_ktc,sr_noisematrix]
#nonlinearity
A_SNratio=0.05
A_SFratio=1.05
ADCratio=1.1
nonlinearity=[A_SNratio,A_SFratio,ADCratio]

class Cccd:
	illumination=[input_screen_size,input_screen_hole_size,input_screen_blur,amplitude_coeff]
	lambda1 = 550*nm  #optical wavelength
	#<--Select (uncomment) the type of a photo sensor
	#SensorType = 'CCD'	
	SensorType = 'CMOS'
	pixel_size = [5*mum, 5*mum] #pixels size
	t_I  = 1*10**(-2) #Exposure/Integration time
	QE_I = 0.8	  #Quantum Efficiency of the photo sensor
	FillFactor    = 0.5#Pixel Fill Factor for CMOS photo sensors.
	QuantumYield  = 1  #quantum yield (number of electrons per one photon interaction)
	FW_e  = 2*10**4    #full well of the pixel
	V_REF = 3.1        #Reference voltage to reset the sense node(3-10V)
	#<--END :: General parameters of the photosensor
	#<--Start :: Sense Nose 
	A_SN  = 5*10**(-6) #Sense node gain, A_SN [V/e]
	A_SF   = 1         #Source follower gain, [V/V]
	A_CDS  = 1         #Correlated Double Sampling gain, [V/V]
	N_bits = 12        #noise is more apparent on high Bits
	S_ADC_OFFSET = 0   #Offset of the ADC, in DN
	flag=[photonshotnoise,fPRNU,darkcurrent,darkcurrent_Dshot,darkcurrent_DarkFPN_pixel,darkcurrent_offsetFPN,sensenoderesetnoise,plots,electrons,volts,fDN,darkframe,Venonlinearity,VVnonlinearity,ADCnonlinearity]
	noise=[nPRNU,darkFPN,darkFPN_offset,sn_reset]
	#<--Start:: Dark Current Noise parameters
	T = 300  #operating temperature, [K]
	DFM = 1  #dark current figure of merit
	Eg_0= 1.1557 #Sensor material constants 
	alpha= 7.021*10**(-4) # material parameter, [eV/K]
	beta= 1108            # material parameter, [K]
	h = 6.62606896*10**(-34) #Plank's constant, in [Joule*s]
	c = 2.99792458*10**8     #speed of light, in [m/s]
	Boltzman_Constant= 8.617343*10**(-5) #Boltzman constant, [eV/K].
	Boltzman_Constant_JK  = 1.3806504*10**(-23) #Boltzman constant, [J/K].
	q 	= 1.602176487*10**(-19)   #a charge of an electron [C], Cylon
	k1	= 10.909*10**(-15)

def tool_circ(x,y,D):
	r=np.sqrt(x**2+y**2)
	z=(r<D/2).astype(np.float)
	z[r==D/2]=0.5
	return z

def ccd_illumination_prepare(Cccd,N,M):
	delta_x=Cccd.illumination[0]*1.0/N
	delta_y=Cccd.illumination[0]*1.0/M
	dx=np.arange(-N/2*delta_x,N/2*delta_x,delta_x)
	dy=np.arange(-N/2*delta_y,N/2*delta_y,delta_y)
	[x1,y1]=np.meshgrid(dx,dy)
	Uin = Cccd.illumination[3] * tool_circ(x1, y1, Cccd.illumination[1])
	#if Cccd.flag[8][0]==1:
	#	Uin_irradiance=np.abs(Uin)**2
	"""
	import matplotlib.pyplot as plt
	plt.imshow(Uin)
	plt.colormap()	
	plt.show()
	"""
	return Uin
def tool_rand_distributions_generator(distribName, distribParams, sampleSize):
	funcName ='tool_rand_distributions_generator'
	distribNameInner = distribName.lower()
	out = []
	if np.prod(sampleSize)>0:
		if distribNameInner in {'exp','exponential'}:
			checkParamsNum(funcName,'Exponential','exp',distribParams,[1]) 
			lambda2  = distribParams[0]
			validateParam(funcName,'Exponential','exp','lambda','lambda',lambda2,{'> 0'})
			out = -np.log(np.random.rand(sampleSize))/lambda2
		elif distribNameInner in {'lognorm','lognormal','cobbdouglas','antilognormal'}:
			checkParamsNum(funcName, 'Lognormal', 'lognorm', distribParams, [0, 2]);
			if len(distribParams)==2:
				mu  = distribParams[0]
				sigma  = distribParams[1]
				validateParam(funcName,'Lognormal','lognorm','[mu,sigma]', 'sigma',sigma,{'> 0'})
			else:
				mu = 0
				sigma = 1
			out = np.exp(mu +sigma*np.ramdom.randn( sampleSize))
		elif distribNameInner in {'ig', 'inversegauss', 'invgauss'}:
			checkParamsNum(funcName,'Inverse Gaussian','ig',distribParams,[2])
			theta = distribParams[0]
			chi = distribParams[1]
			validateParam(funcName,'Inverse Gaussian','ig','[theta,chi]','theta', theta, {'> 0'})
			validateParam(funcName, 'Inverse Gaussian', 'ig', '[theta, chi]', 'chi', chi, {'> 0'})

			chisq1 = np.random.randn(sampleSize)**2;
			out = theta + 0.5*theta/chi * ( theta*chisq1-np.sqrt(4*theta*chi*chisq1 + theta**2*chisq1**2) )
			l = np.random.rand(sampleSize) >= theta/(theta+out)
			out[l] = theta**2/out[l]

		elif distribNameInner in {'logistic'}:
			checkParamsNum(funcName, 'Logistic', 'logistic', distribParams, [0, 2]);
			if len(distribParams)==2:  #numel
  				a  = distribParams[0]
				k  = distribParams[1]
				validateParam(funcName, 'Laplace', 'laplace', '[a, k]', 'k', k, {'> 0'})
			else:
				a = 0
				k = 1
			u1 = np.random.rand(sampleSize)
			out = a -k*np.log( 1/u1 -1)	
		elif distribNameInner in {'wald'}:
			checkParamsNum(funcName, 'Wald', 'wald', distribParams, [1])
			chi = distribParams[0]
			validateParam(funcName, 'Wald', 'wald', 'chi', 'chi', chi, {'> 0'})
			out = tool_rand_distributions_generator( 'ig', [1,chi], sampleSize)
		else:
			print '\nRANDRAW: Unknown distribution name: ', distribName
	return out
def  checkParamsNum(funcName, distribName, runDistribName, distribParams, correctNum):
	#感觉没有意义，尊重原创
	if ~any( len(distribParams) == correctNum):
		print distribName,'Variates Generation:\n','Wrong numebr of parameters (run',funcName,'(',runDistribName,') for help)'
	return
def validateParam(funcName,distribName,runDistribName,distribParamsName,paramName,param,conditionStr):
	condLogical = 1
	eqCondStr = []
	for nn in range(0,len(conditionStr)):
		if nn==1:
			eqCondStr = [eqCondStr or conditionStr]
     		else:
          		eqCondStr = [eqCondStr and conditionStr]         
     		eqCond = conditionStr[0]
     		if eqCond=={'<'}:
			condLogical = condLogical & (param< float(conditionStr[2:]))
     		elif eqCond=={'<='}:
			condLogical = condLogical & (param<=float(conditionStr[2:]))              
     		elif eqCond=={'>'}:
			condLogical = condLogical & (param> float(conditionStr[2:])) 
     		elif eqCond=={'>='}:
			condLogical = condLogical & (param>=float(conditionStr[2:]))
     		elif eqCond=={'~='}:
			condLogical = condLogical & (param!=float(conditionStr[2:]))
     		elif eqCond=={'=='}:
			if cmp(conditionStr[2:],'integer')==0:
				condLogical = condLogical & (param==np.floor(param))          
			else:
				condLogical = condLogical & (param==float(conditionStr[2:]))
	if condLogical==0:
		print distribName,'Variates Generation:tool_rand_distributions_generator,(',runDistribName,distribParamsName,'SampleSize)\n Parameter paramName should be eqCondStr\n (run',funcName,'(',runDistribName,')for help)'
	return            
def ccd_set_photosensor_constants(Uin,Cccd):
	Cccd.Signal_CCD_photons= np.zeros([fN,fM])
	Cccd.Signal_CCD_electrons= np.zeros([fN,fM])
	Cccd.dark_signal= np.zeros([fN,fM])
	Cccd.nonlinearity=[A_SNratio,A_SFratio,ADCratio]
	Cccd.sensor_size=[fN,fM]
	Cccd.light_signal= np.zeros([fN,fM])
	Cccd.Signal_CCD_voltage= np.zeros([fN,fM])
	Cccd.Signal_CCD_DN= np.zeros([fN,fM])
	return Cccd
def ccd_FPN_models(Cccd, sensor_signal_rows, sensor_signal_columns, noisetype, noisedistribution, noise_params):
	if noisedistribution=='AR-ElGamal':
		if cmp(noisetype,'pixel')==0:
			x2=np.random.randn(sensor_signal_rows,sensor_signal_columns)
			noiseout=signal.lfilter([1,0],noise_params,x2)	
		elif cmp(noisetype, 'column')==0:
			x=signal.lfilter([1,0],noise_params, np.random.randn(1,sensor_signal_columns))
			noiseout=repmat(x,sensor_signal_rows,1)
	elif noisedistribution=='Janesick-Gaussian':
		if cmp(noisetype, 'pixel')==0:
			noiseout = np.random.randn(sensor_signal_rows,sensor_signal_columns)	
		elif cmp(noisetype, 'column')==0:
			x = np.random.randn(1,sensor_signal_columns)
			noiseout = repmat(x,sensor_signal_rows,1)
		elif cmp(noisetype, 'row')==0:
			x = np.random.randn(sensor_signal_rows,1)
			noiseout = repmat(x,1,sensor_signal_columns)	
	elif noisedistribution=='Wald':
		if cmp(noisetype,'pixel')==0:
			noiseout = tool_rand_distributions_generator('wald',noise_params[0],[sensor_signal_rows, sensor_signal_columns]) + np.random.rand(sensor_signal_rows, sensor_signal_columns)
		elif cmp(noisetype, 'column')==0:
			x = tool_rand_distributions_generator('lognorm',[noise_params[0],noise_params[1]],[1, sensor_signal_columns])
			noiseout = repmat(x,sensor_signal_rows,1)
	return noiseout

def ccd_photosensor_lightFPN(Cccd):
	#Cccd.noise[0][3]  = ccd.noise.nPRNU.n_noisematrix---
	#Cccd.noise[0][0]) = ccd.noise.nPRNU.model
	#Cccd.noise[0][1]) = ccd.noise.nPRNU.nparameters
	Cccd.noise[0][3]=ccd_FPN_models(Cccd,Cccd.sensor_size[0],Cccd.sensor_size[1],'pixel',Cccd.noise[0][0],Cccd.noise[0][1])
	Cccd.light_signal=Cccd.light_signal*(1+Cccd.noise[0][3]*Cccd.noise[0][2])
	return 	Cccd.light_signal

def ccd_photosensor_lightnoises(Cccd,Uin):
	if cmp('CMOS',Cccd.SensorType)==0:
		PA=Cccd.FillFactor*Cccd.pixel_size[0]*Cccd.pixel_size[1]
	else:
		PA=Cccd.pixel_size[0]*Cccd.pixel_size[1]
	Uin_irradiance = PA*abs(Uin)**2
	P_photon = (Cccd.h * Cccd.c)/ Cccd.lambda1
	Cccd.Signal_CCD_photons = np.round(Uin_irradiance * Cccd.t_I / P_photon)
	if Cccd.flag[0]==1:
		Cccd.Signal_CCD_photons=Cccd.Signal_CCD_photons-np.min(Cccd.Signal_CCD_photons)
		Cccd.Signal_CCD_photons=np.random.poisson(Cccd.Signal_CCD_photons)
	QE=Cccd.QE_I*Cccd.QuantumYield
	Cccd.light_signal = Cccd.Signal_CCD_photons*QE
	if Cccd.flag[1]==1:
		Cccd.light_signal=ccd_photosensor_lightFPN(Cccd)
	return Cccd.light_signal

def ccd_photosensor_darkFPN(Cccd):
	Cccd.noise[1][3] = ccd_FPN_models(Cccd, Cccd.sensor_size[0], Cccd.sensor_size[1], 'pixel', Cccd.noise[1][1], Cccd.noise[1][3])
	Cccd.dark_signal = Cccd.dark_signal*(1 + (Cccd.noise[1][0])*(Cccd.noise[1][3]))
	return Cccd.dark_signal

def ccd_photosensor_darknoises(Cccd):
	PA=Cccd.pixel_size[0]*Cccd.pixel_size[1]*10**4
	Cccd.Eg=Cccd.Eg_0-(Cccd.alpha*(Cccd.T**2))/(Cccd.beta+Cccd.T)
	Cccd.DARK_e = (Cccd.t_I)*2.55*10**15*PA*Cccd.DFM*(Cccd.T**(1.5))*np.exp(-Cccd.Eg/(2*Cccd.Boltzman_Constant*Cccd.T))
	Cccd.dark_signal = (Cccd.DARK_e)*np.ones([len(Cccd.Signal_CCD_electrons),len(Cccd.Signal_CCD_electrons[0])])
	#<----- ### Start:: adding Dark Shot noise
	if Cccd.flag[3]==1:   #Cccd.glag[3]=ccd.flag.darkcurrent_Dshot
		Cccd.dark_signal=np.random.poisson(Cccd.dark_signal) 
	#<----- ### END:: adding Dark Shot noise
	#<----- ### Start:: adding Dark FPN  %%% being added to dark current, it is too small.
	if Cccd.flag[4]==1:   #Cccd.flag[4]=Cccd.flag.darkcurrent
		Cccd.dark_signal=ccd_photosensor_darkFPN(Cccd)
	#<----- ### END:: adding Dark FPN  %%% being added to dark current, it is too small.
	return Cccd.dark_signal
#<----- ### Start:: Sense Node - charge-to-voltage conversion
def ccd_sense_node_chargetovoltage(Cccd):
	Cccd.C_SN=Cccd.q/Cccd.A_SN
	Cccd.V_FW=Cccd.FW_e*Cccd.q/Cccd.C_SN
	Cccd.V_min=Cccd.q*Cccd.A_SN/Cccd.C_SN
	if cmp('CMOS',Cccd.SensorType)==0:
		if Cccd.flag[6]==1:  #Cccd.flag[6]=ccd.flag.sensenoderesetnoise
			if Cccd.noise[3][0]>1: #Cccd.noise[3][0]=ccd.noise.sn_reset.Factor
				Cccd.noise[3][0]=1
				print 'Sensor Simulator::: Warning! The compensation factor you entered',Cccd.noise[3][0],' for \n the Sense Node Reset Noise cannot be more than 1! The factor is set to 1.\n'
			elif Cccd.noise[3][0]<0:
				Cccd.noise[3][0]=0
				print 'Sensor Simulator::: Warning! The compensation factor you entered ',Cccd.noise[3][0],'for the Sense Node Reset Noise cannot be negative! The factor is set to 0, SNReset noise is not simulated.'
			Cccd.noise[3][1]=np.sqrt((Cccd.Boltzman_Constant_JK)*(Cccd.T)/(Cccd.C_SN))
			Cccd.noise[3][2]=np.exp(Cccd.noise[3][1]*np.random.randn(Cccd.sensor_size[0],Cccd.sensor_size[1] ))-1
			if Cccd.flag[12]==1:             
				Cccd.Signal_CCD_voltage =(Cccd.V_REF+Cccd.noise[3][0]*Cccd.noise[3][2])*(np.exp(-Cccd.nonlinearity[0]*Cccd.q*Cccd.Signal_CCD_electrons/Cccd.k1))
			else:
				Cccd.Signal_CCD_voltage=(Cccd.V_REF+ Cccd.noise[3][0]*Cccd.noise[3][2])-(Cccd.Signal_CCD_electrons*Cccd.A_SN)
		else:
			if Cccd.flag[12]==1:
				Cccd.Signal_CCD_voltage=Cccd.V_REF*(np.exp(-Cccd.nonlinearity[0]*Cccd.q*Cccd.Signal_CCD_electrons/Cccd.k1))
			else:
				Cccd.Signal_CCD_voltage = Cccd.V_REF- (Cccd.Signal_CCD_electrons*Cccd.A_SN)
	else: #<---The sensor is CCD
		if Cccd.flag[12]==1:
			Cccd.Signal_CCD_voltage=Cccd.V_REF*(np.exp(-Cccd.nonlinearity[0]*Cccd.q*Cccd.Signal_CCD_electrons/Cccd.k1))
		else: 
			Cccd.Signal_CCD_voltage=Cccd.V_REF-Cccd.Signal_CCD_electrons*Cccd.A_SN
	return Cccd.Signal_CCD_voltage
def ccd_source_follower(Cccd):
	if Cccd.flag[13]==1: #Cccd.flag[10]=ccd.flag.VVnonlinearity
		nonlinearity_alpha = (Cccd.A_SF*(Cccd.nonlinearity[1]-1))/(Cccd.V_FW)
		Cccd.A_SF_new= nonlinearity_alpha*((Cccd.V_REF-Cccd.Signal_CCD_voltage)/(Cccd.V_REF))+(Cccd.A_SF)*np.ones([Cccd.sensor_size[0],Cccd.sensor_size[1]])
		Cccd.Signal_CCD_voltage=(Cccd.Signal_CCD_voltage)*(Cccd.A_SF_new)
	else:
		Cccd.Signal_CCD_voltage=Cccd.Signal_CCD_voltage*Cccd.A_SF
	return Cccd.Signal_CCD_voltage
def ccd_cds(Cccd):
	if cmp('CMOS',Cccd.SensorType)==0:
		if Cccd.flag[5] == 1:  #ccd.flag[5]=ccd.flag.darkcurrent_offsetFPN
			Cccd.noise[2][3]=ccd_FPN_models(Cccd,Cccd.sensor_size[0], Cccd.sensor_size[1],'column',Cccd.noise[2][0], Cccd.noise[2][1])
			Cccd.Signal_CCD_voltage = Cccd.Signal_CCD_voltage*(1+ Cccd.noise[2][3]*(Cccd.V_FW*Cccd.noise[2][2]))
	Cccd.Signal_CCD_voltage=Cccd.Signal_CCD_voltage*Cccd.A_CDS
	return Cccd.Signal_CCD_voltage
def ccd_adc(Cccd):
	N_max=2**Cccd.N_bits
	Cccd.A_ADC=N_max/(Cccd.V_FW-Cccd.V_min)
	if Cccd.flag[14]==1:    #Cccd.flag[14]=ccd.flag.ADCnonlinearity
		A_ADC_NL=Cccd.nonlinearity[2]*Cccd.A_ADC
		nonlinearity_alpha=(np.log(A_ADC_NL)/np.log(Cccd.A_ADC)-1)/Cccd.V_FW
		signal=Cccd.V_REF-Cccd.Signal_CCD_voltage
		A_ADC_new = (Cccd.A_ADC)*np.ones([len(signal),len(signal[0])])
		A_ADC_new = A_ADC_new**(1-nonlinearity_alpha*signal)
		S_DN = np.round(Cccd.S_ADC_OFFSET+A_ADC_new*signal)
	else:
		S_DN = np.round(Cccd.S_ADC_OFFSET+Cccd.A_ADC*(Cccd.V_REF-Cccd.Signal_CCD_voltage))
	S_DN[S_DN<=0]=0
	S_DN[S_DN>=N_max]=N_max
	Cccd.Signal_CCD_DN=S_DN
	return Cccd.Signal_CCD_DN

def ccd_photosensor(Uin,Cccd):
	Cccd=ccd_set_photosensor_constants(Uin,Cccd)
	if Cccd.flag[11]!=1:   # Cccd.flag[11]=ccd.flag.darkframe
		#<-This routine for adding light noise (photon shot noise and photo response non-uniformity
		Cccd.light_signal=ccd_photosensor_lightnoises(Cccd, Uin)
	if Cccd.flag[2] ==1:  # Cccd.flag[2]=ccd.flag.darkcurrent
		#<-adding dark current noises that consist of Dark FPN and Dark shot noise
		Cccd.dark_signal=ccd_photosensor_darknoises(Cccd)

	Cccd.Signal_CCD_electrons = Cccd.light_signal + Cccd.dark_signal
	idx = (Cccd.Signal_CCD_electrons>=Cccd.FW_e)
	Cccd.Signal_CCD_electrons[idx] = Cccd.FW_e
	Cccd.Signal_CCD_electrons=np.floor(Cccd.Signal_CCD_electrons)

	Cccd.Signal_CCD_voltage = ccd_sense_node_chargetovoltage(Cccd)
	#<-- Signal's Voltage amplification by Source Follower
	Cccd.Signal_CCD_voltage = ccd_source_follower(Cccd)
	#<-- Signal's amplification and de-noising by Correlated Double Sampling
	Cccd.Signal_CCD_voltage = ccd_cds(Cccd)
	#<-- Analogue-To-Digital Converter
	Cccd.Signal_CCD_DN = ccd_adc(Cccd)
	return Cccd
#<--------------------------------->#
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
#<--------------------main------------>
if __name__ == '__main__':
	'''
    	if Cccd.flag[11]==0:
		Uin=ccd_illumination_prepare(Cccd,N,M)
	else:
		Uin=np.zeros(N)
    	'''        
	hh=fits.open('/home/awen/桌面/CCD_modpicture/NGC3231_2016_02_03_14_55_27_145.fits')
	Uin=hh[0].data
	Uin=Uin.astype(np.float64)
	Uin=(Uin-np.min(Uin))/np.max(Uin)  #归一化到0-1之间
	fM=len(Uin)
	fN=len(Uin[0])
	Cccd = ccd_photosensor(Uin,Cccd)
	AD=Adnoise()
	f=Cccd.Signal_CCD_DN
	f=AD.adcloud(f, random.uniform(-10,10)) #在[-10,10]取一个随机数作为阈值
	f=AD.adcosmic_rays(f,5)
	f=AD.adline(f,5)
	show = fits.PrimaryHDU(f)
	showlist = fits.HDUList([show])
	showlist.writeto('/home/awen/桌面/CCD_modpicture/CMOS_Signal_CCD_DN_py001.fits')












