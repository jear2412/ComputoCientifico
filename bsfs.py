#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:40:59 2019

@author: Javier
@title: Backward and Forward substitution
"""

#Backward substitution
def BS(T, b, y):
    n=len(y)
    y[n-1]=b[n-1]/T[n-1][n-1]
    for i in range(n-2,-1,-1):
        s=0
        for j in range(i, n,1):
            s=s+T[i][j]*y[j]
        y[i]=1/T[i][i]*(b[i]-s)
    
#Forward substitution
def FS(L,b, y):
    n=len(y)
    y[0]=b[0]/L[0][0]
    for i in range(1,n,1):
        s=0
        for j in range(0, i,1):
            s=s+L[i][j]*y[j]
        y[i]=1/L[i][i]*(b[i]-s)