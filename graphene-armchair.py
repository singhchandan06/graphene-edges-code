from numpy import *
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
from numpy import linalg as LA
import random
import math 

import time
import glob
import numpy as np


def graphene_arm(t,kx,Nhex):
    
    phas = complex(cos(kx),sin(kx))
    Ham =  zeros([4*Nhex+6,4*Nhex+6],dtype='complex')
    
    for i in arange(Nhex+2):
        Ham[4*i,4*i+1] = t; Ham[4*i+1,4*i] = t
        
    
    #for edge hopping
    
    for i in arange(Nhex+1):
        Ham[4*i+2, 4*i+3] = t*phas; Ham[4*i+3,4*i+2] = t/phas
        
    # for x bond    
    for i in arange(Nhex+1):
        Ham[4*i,4*i+2] = t;   Ham[4*i+2, 4*i] = t   
        Ham[4*i+3,4*i+5] = t; Ham[4*i+5,4*i+3] = t
        
    # for y bond    
    for i in arange(Nhex+1):
        Ham[4*i+2,4*i+4] = t;   Ham[4*i+4,4*i+2] = t
        Ham[4*i+1,4*i+3] = t; Ham[4*i+3,4*i+1] = t
    
    return Ham

t = 1
nk = 5001
Nhex = 4

kx0 = linspace(-pi,pi,nk)
kk = []
ssp = zeros([4*Nhex+6,nk])

for i in arange(nk):
    kx = kx0[i];
    kk.append(kx)
    
    Hamil = graphene_arm(t,kx,Nhex)
    evals, evecs = LA.eigh(Hamil)
    evals = sorted(evals)
    ssp[:,i] = evals

    
fig, ax = plt.subplots()

for j in np.arange(4*Nhex+6):    ## to plot multiple plot together 
    		plt.plot(kk, ssp[j,:],color=((1.*j)/(4*Nhex+6), 0.2, 0.2),  lw=1.5)

ax.set_ylabel("Energy (eV)", fontsize=20)
        
fig.tight_layout()
fig.savefig("band-armchair.pdf")
print('Done.\n')
