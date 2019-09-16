#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:43:34 2019

@author: Javier
@title: Cholesky Factorization
"""
import numpy as np

#CHOLESKY FACTORIZATION
#Obs: Saves in the upper triangular part of A
            
            
def Chol(A):
    
    n=np.shape(A)[0]
    for k in range(0,n):
        s=0
        for j in range(0,k):
            s+=(A[j][k]**2)
        A[k][k]=  math.sqrt( A[k][k]-s  )
        for i in range(k+1, n):
            s=0
            for j in range(0, k):
                s+= A[j][i]*A[j][k]
            A[k][i]=1/A[k][k]*( A[k][i]-s  )          
        

