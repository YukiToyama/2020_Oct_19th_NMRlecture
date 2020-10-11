# -*- coding: utf-8 -*-
"""
This script allows you to simulate the free protomer and bound protomer populations,
using the P + PL -> PL model (monomer model) or P2 + L -> P2L + P2L -> P2L2 (dimer model)
In dimer model, several different values for the cooperative factor, alpha, are assumed.

2020/9/21
Y. Toyama
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize as opt
import scipy.fftpack as fftpack
from functools import partial

## Parameter set ####
# Here we assume the total number of the binding site is the same between monomer and dimer cases.
 
PT = 200E-6 # M, monomer protein concentration
PTd = PT*0.5 # M, dimer protein concentration. 

K=20000 # M-1
alpha = np.array([5,1,0.2])
conc = np.arange(0,1000E-6,1E-6)

#  Plot
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.rcParams["font.family"] = "Arial"
colorlist=["tomato","navy","lightskyblue","deeppink"]

outname = "dimermodel_slow_population"

## Monomer model
def monomer(LT,PT,K):
  L = (-1*(PT-LT+K**-1)+np.sqrt((PT-LT+K**-1)**2+4*LT*K**-1))/2
  PL = LT-L
  P = PT-LT+L
  return P, PL, L

P = np.empty_like(conc,dtype=float)
PL = np.empty_like(conc,dtype=float)
Lm = np.empty_like(conc,dtype=float)

for i in range(len(conc)):
  P[i],PL[i],Lm[i]=monomer(conc[i],PT,K)

UP= P/PT
UPL= PL/PT

ax1.plot(1E6*conc,UP,color='navy',linewidth=2, ls='--')   
ax1.plot(1E6*conc,UPL,color='navy',linewidth=2, ls='-')  
ax1.yaxis.major.formatter._useMathText = True
ax1.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
ax1.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=16)
ax1.set_ylabel('Population',fontsize=16)
ax1.set_xlabel('Ligand concentration ($\mu$M)',fontsize=16)
ax1.spines[ 'top' ].set_linewidth( 1 )
ax1.spines[ 'left' ].set_linewidth( 1 )
ax1.spines[ 'right' ].set_linewidth( 1 )
ax1.spines[ 'bottom' ].set_linewidth( 1 )
               
## Dimer model

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
  
P2 = np.empty_like(conc,dtype=float)
P2L = np.empty_like(conc,dtype=float)
P2L2 = np.empty_like(conc,dtype=float)
Ld = np.empty_like(conc,dtype=float)

for k in range(len(alpha)):
  for i in range(len(conc)):
      p = [PTd,1e-6,1e-6,conc[i]]
      q = [PTd,conc[i],alpha[k],K]
      ffs_partial = partial(ffs_complex,q)
      # Solutions are ordered according to how the initial guess vector is arranged
      P2[i],P2L[i],P2L2[i],Ld[i]=opt.root(ffs_partial,p,method='lm').x
      
  UP2 = 2*P2/(2*PTd)
  UP2L = P2L/(2*PTd)
  BP2L = P2L/(2*PTd)
  BP2L2 = 2*P2L2/(2*PTd)
  
  ax2.plot(1E6*conc,UP2+UP2L,color=colorlist[k],linewidth=2, ls='--')   
  ax2.plot(1E6*conc,BP2L+BP2L2,color=colorlist[k],linewidth=2, ls='-',label=r'$\alpha$ = '+str(alpha[k]))  


ax2.yaxis.major.formatter._useMathText = True
ax2.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
ax2.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=16)
ax2.set_ylabel('Population',fontsize=16)
ax2.set_xlabel('Ligand concentration ($\mu$M)',fontsize=16)
ax2.legend(loc='center right',fontsize=16,frameon=True)
ax2.spines[ 'top' ].set_linewidth( 1 )
ax2.spines[ 'left' ].set_linewidth( 1 )
ax2.spines[ 'right' ].set_linewidth( 1 )
ax2.spines[ 'bottom' ].set_linewidth( 1 )

plt.tight_layout()
plt.savefig(outname+'.pdf')

