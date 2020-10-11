# -*- coding: utf-8 -*-
"""
This script allows you to simulate the NMR spectra of the ligand titartion experiments to a symmetric, dimeric receptor protein,
using the P2 + L -> P2L + L -> P2L2 (dimer model).
Here, I assume the positive/negative cooperativity only affects the koff value.

2020/9/21
Y. Toyama

"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize as opt
import scipy.fftpack as fftpack
from functools import partial
from cycler import cycler


# Parameter set 
file = "dimerbinding_1"
PT = 100E-6 # M
alpha = 5
K=20000 # microscopic affinity, M^-1

LtoPlist = np.array([0.,0.1,0.3,0.5,1,3,5,10])  # LT/PT ratios
Llist=LtoPlist*PT

kp1=5E6
km1=kp1/K

kp2=kp1
km2=kp2/(K*alpha)

wUP2=0*2*np.pi
wUP2L=0*2*np.pi
wBP2L=200*2*np.pi
wBP2L2=200*2*np.pi

RUP2=RUP2L=RBP2L=RBP2L2=10

# Acq parameters
TD=1024
SW=2000	#Hz

sampling=np.power(SW*2,-1.)
Tmax=sampling*TD
t = np.arange(0,Tmax,sampling)

#Apodization param
SPoff=0.5
SPend=1.0
SPpow=2

# Cosine apodization (following nmrPipe notation)
def SP(FID,off,end,power):
    APOD = np.empty_like(FID)
    tSize = len(FID)
    for i in range(len(FID)):
        APOD[i] = FID[i]*np.sin(np.pi*off + np.pi*(end-off)*i/(tSize-1) )**power
    return APOD
  
  
# Define the function to solve the thermodynamic equation
def ffs_complex(q,p):
    
    # Unpack variables and constants
    P2, P2L, P2L2, L = p # Variables
    PT, LT, alpha, K = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -PT + P2 + P2L + P2L2 # Protein equation
    eq2 = -LT + L + P2L + 2*P2L2 # Ligand equation
    eq3 = 2*K*P2*L - P2L
    eq4 = (1/2)*alpha*K*P2L*L - P2L2
    
    return [eq1, eq2, eq3, eq4]

  
#  Plot the overlaied spectra
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.rcParams["font.family"] = "Arial"
plt.rcParams['axes.prop_cycle']  = cycler(color=['navy','slateblue','mediumorchid','lightseagreen',
                                                  'limegreen','orange','tomato','firebrick'])

for i in range(len(Llist)):
  LT = Llist[i]
  # Calculate the P2, P2L, P2L2, and L concentrations.
  p = [PT,1e-6,1e-6,LT]
  q = [PT,LT,alpha,K]
  ffs_partial = partial(ffs_complex,q)
  P2,P2L,P2L2,L=opt.root(ffs_partial,p,method='lm').x
  
  # Define the relaxation matrix
  R = np.zeros((4,4),dtype=complex)
  R[0,0]=-2*kp1*L+wUP2*1j-RUP2
  R[0,1]=km1
  R[0,2]=km1
  R[0,3]=0
  R[1,0]=kp1*L
  R[1,1]=-1*km1-kp2*L+wUP2L*1j-RUP2L
  R[1,2]=0
  R[1,3]=km2
  R[2,0]=kp1*L
  R[2,1]=0
  R[2,2]=-1*km1-kp2*L+wBP2L*1j-RBP2L
  R[2,3]=km2
  R[3,0]=0
  R[3,1]=kp2*L
  R[3,2]=kp2*L
  R[3,3]=-2*km2+wBP2L2*1j-RBP2L2
  
  # Degine the initial magnetization
  I = np.zeros(4,dtype=complex)
  I[0]=2*P2/(2*PT)
  I[1]=P2L/(2*PT)
  I[2]=P2L/(2*PT)
  I[3]=2*P2L2/(2*PT)
  
  #caluculate FID
  def FID(t):
    It=np.dot(sp.linalg.expm(R*t),I)
    return It[0]+It[1]+It[2]+It[3]
  
  FIDv = np.vectorize(FID)
  
  FIDapod=SP(FIDv(t),SPoff,SPend,SPpow)
  S=fftpack.fftshift(fftpack.fft(FIDapod,n=8*TD))
  F=fftpack.fftshift(fftpack.fftfreq(8*TD,sampling))
  
  ax.plot(F,np.real(S),linewidth=2)   

ax.yaxis.major.formatter._useMathText = True
ax.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=16)
ax.set_ylabel('Intensity',fontsize=16)
ax.set_xlabel('Frequency (Hz)',fontsize=16)
ax.set_xlim(-50,250)
#ax.legend(loc='upper right',fontsize=12,frameon=True)

ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 1 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 1 )

plt.tight_layout()
plt.savefig(file+".pdf")
