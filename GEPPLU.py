#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:41:59 2019

@author: Javier
@title: Gaussian Elimination with Partial Pivoting
"""

import numpy as np

#Special forward substitution for when diag L={1}
#This is important since we store L within A in PALU
def FSL(L,b, y):
    n=len(y)
    y[0]=b[0]
    for i in range(1,n,1):
        s=0
        for j in range(0, i,1):
            s=s+L[i][j]*y[j]
        y[i]=b[i]-s




#Finds the maximum of a given vector and returns its position
        
def MaxIndex(x):
    a=np.amax( np.abs(x) )
    for i in range(0,len(x)):
        if(x[i]==a or x[i]==-a ):
            return i
    
    
#Gaussian Elimination with Partial Pivoting
#Obs: It saves the multipliers and U over A
#U is the upper part and the multipliers the lower part
def GEPP(A):
    n=np.shape(A)[0]
    r=np.zeros(n-1) #permutation vector
    for k in range(0, n-1):
        temp=MaxIndex( A[:,k][k:n])
        r[k]=temp+k
        
        #exchange of rows
        for j in range(k,n):
            temp=A[k][j]
            A[k][j]=A[int(r[k])][j]
            A[int(r[k])][j]=temp
        #save multipliers
        for i in range(k+1, n):
            A[i][k]=-A[i][k]/A[k][k]
        
        #change rows
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[i][j]=A[i][j]+A[i][k]*A[k][j]
            
    return(r)  

#Permutates the multipliers according to p and
#saves L in the lower part of A 
def GEPPLU(A, p):
    #p is a permutation vector
    n=np.shape(A)[0]
    for k in range(0, n-1):
        for i in range(k+1, n-1):
            temp= A[i][k]
            A[i][k]= A[int(p[i])][k]
            A[int(p[i])][k]= temp
            
        for i in range(k+1, n):
            A[i][k]=-A[i][k]

            
#Returns L and modifies it to have a diagonal of 1s
            
def LGEPP(A):
    B=A.copy()        
    B=np.tril(B)
    for i in range(0, np.shape(B)[0]):
        B[i][i]=1
    return B    

#Returns U in GEPP
def UGEPP(A):
    return np.triu(A)
        
#permutates vector b in Ax=b according to p
      
def Pb(p,b):
    n=len(b)
    for i in range(0, n-1):
        temp=b[i]
        b[i]=b[int(p[i])]
        b[int(p[i])]=temp
     
