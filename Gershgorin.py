#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:35:00 2019

@author: Javier
Title: Gershgorin Circles
Draw Gershgoriin Circles
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection




def G(A):
    n=len(A)
    eigen=np.linalg.eig(A)[0]
    patches=[]
    for i in range(n):
        h = np.real(A[i,i])
        k = np.imag(A[i,i])
        r = np.sum(np.abs(A[i,:])) - np.abs(A[i,i]) 
        circle = Circle((h, k), r) #center h,k radius r
        patches.append(circle)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = 100*np.random.random(100)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    p.set_array(colors)
    ax.add_collection(p)
    plt.axis('equal')    
    plt.title('Gershgorin Circles')
    for h, k in zip(np.real(eigen), np.imag(eigen)):
        plt.plot(h, k,'o')
    #p.set_clim([5, 50])
    plt.savefig("gcircles", dpi = 600)
    plt.show()

#Example
A=np.array([  [8.,1,0],[1,4,5], [0,5,1]    ])
G(A)