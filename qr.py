#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:55:41 2019

@author: Javier
@title: QR
"""

import numpy as np

# QR factorization of A
# A=QR        
#Rewrites Q over A

def QR(A):
    n=np.shape(A)[1]
    R=np.zeros( [n,n] )        
    for i in range(0,n):
        R[i,i]=np.linalg.norm(A[:,i])
        A[:,i]=A[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j]=np.dot(A[:,i], A[:,j])
            A[:,j]=A[:,j]-R[i,j]*A[:,i]
    return(R)    
