from numpy import *
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
from numpy import linalg as LA
import random
import math 

import time
%matplotlib inline
import glob
import numpy as np


def graphene_zig(t,t2,kx,N):
    
    phas = complex(cos(kx),sin(kx))
    Ham =  zeros([2*N+2,2*N+2],dtype='complex')
    Ham2 = zeros([2*N+2,2*N+2],dtype='complex')
    
    
    for i in arange(int(N/2)+1):  #(0,1), (4,5), (8,9),...
        Ham[4*i,4*i+1] = t + t/phas; Ham[4*i+1,4*i] = t + t*phas
        
    for i in arange(int(N/2)):    #(2,3), (6,7),...
        Ham[4*i+2,4*i+3] = t + t*phas; Ham[4*i+3,4*i+2] = t + t/phas
        
    for i in arange(N):     #(1,2), (3,4), (5,6),...
        Ham[2*i+1,2*i+2] = t ; Ham[2*i+2,2*i+1] = t
        
# for SOC
    for i in arange(N+1):     
        Ham[2*i,2*i] = t2*phas - t2/phas    #(0,0), (2,2), (4,4),...
        Ham[2*i+1,2*i+1] = -t2*phas + t2/phas    #(1,1), (3,3), (5,5),...
        
    for i in arange(int(N/2)):    
        Ham[4*i,4*i+2] = -t2 + t2/phas; Ham[4*i+2,4*i] = Ham[4*i,4*i+2].T.conj()   #(0,2), (4,6),...
        Ham[4*i+1,4*i+3] = -t2 + t2*phas; Ham[4*i+3,4*i+1] = Ham[4*i+1,4*i+3].T.conj()   #(1,3), (5,7),...
        Ham[4*i+2,4*i+4] = t2 - t2*phas; Ham[4*i+4,4*i+2] = Ham[4*i+2,4*i+4].T.conj()   #(2,4), (6,8),...
        Ham[4*i+3,4*i+5] = t2 - t2/phas; Ham[4*i+5,4*i+3] = Ham[4*i+3,4*i+5].T.conj()   #(3,5), (7,9),...
        
    
    
    return Ham 

t = 1
t2= 0.05
nk =201
N = 40
kx0 = linspace(0,2*pi,nk)
kk = []
ssp = zeros([2*N+2,nk])

for i in arange(nk):
    kx = kx0[i];
    kk.append(kx)
    
    Hamil = graphene_zig(t,t2,kx,N)
    evals, evecs = LA.eigh(Hamil)
    evals = sorted(evals)
    ssp[:,i] = evals

    
fig, ax = plt.subplots()

for j in np.arange(2*N+2):    ## to plot multiple plot together 
    		plt.plot(kk, ssp[j,:],lw=1.5, color='red')

ax.set_ylabel("E/t", fontsize=20)
ax.set_xlabel("$k_x$/a", fontsize=20)
plt.ylim(-1.0,1.0)
        
fig.tight_layout()
fig.savefig("band-zigzag-trivial.pdf")
print('Fig.(1) "Quantum Spin Hall Effect in Graphene" PRL 95, 226801 (2005).\n')
#rint(Hamil[1,2])
