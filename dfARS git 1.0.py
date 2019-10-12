#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:38:52 2019

@author: Javier Enrique Aguilar
@Title: Derivative Free Adaptive Rejection Sampling
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
from scipy import stats
import random
from datetime import datetime
import seaborn
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF



#######Auxiliary Functions

def f(x): #target function
    return x*np.exp(-x) #gamma (2,1)

def h(x): #log of f(x)
    return np.log( f(x)  )


def abss(a,b,x): #function that gives slopes and interceps given x
    for i in range(len(x)-1):
        b[i]=(h(x[i+1])-h(x[i]))/(x[i+1]-x[i])
        a[i]=h(x[i+1])-b[i]*x[i+1]    
               
def ab(x): #slope and intercept given x_i x_i+1
    b=(h(x[1])-h(x[0]))/(x[1]-x[0])
    a=h(x[1])-b*x[1]
    return a,b


#Normalizing constants

def ncExp(a,b,low,up): #normalizing constant of exponential in [xi, x_i+1]
    c=math.exp(a)/b*( math.exp(b*up)-math.exp(b*low) )
    return c


def ncLExp(a1,b1, up):
    c=math.exp(a1)/b1*(math.exp(b1*up )-1)
    return c

def ncRExp(am_1,bm_1,low):
    c= -math.exp(am_1)/(bm_1)*(math.exp(bm_1*low ))
    return c
    


def ncDexp(a,b,x): #normalizing constant of double exponential in [xi, x_i+1]
    i=1
    k=(a[i-1]-a[i+1])/(b[i+1]-b[i-1]) #intersection b/w lines
    #normalizing constant c
    c= math.exp( a[i-1] )/ b[i-1]*( math.exp( b[i-1]*k )-math.exp( b[i-1]*x[i] )   )   
    c=c+math.exp( a[i+1] )/ b[i+1]*( math.exp( b[i+1]*x[i+1] )-math.exp( b[i+1]*k )   )
    return c




def NC(x): #Normalizing constants given the array x
    m=len(x)-1 #number of partitions 
    ncs=np.zeros(m)
    a=np.zeros(m)
    b=np.zeros(m)
    abss(a,b,x)
    for i in range(m):
        if(i==0): #first 
            low=x[i]
            up=x[i+1]
            tempa,tempb=ab(np.array([x[i+1],x[i+2]] ) )
            ncs[i]=ncExp(tempa,tempb,low,up)
        elif(i>0 and i<m-1): #other partitions bw first and last
            tempx=x[i-1:i+2]
            tempa=a[i-1:i+2]
            tempb=b[i-1:i+2]
            ncs[i]=ncDexp(tempa,tempb,tempx)
        elif(i==m-1):
            low=x[i]
            up=x[i+1]
            tempa,tempb=ab(np.array([x[i-1],x[i]] ) )
            ncs[i]=ncExp(tempa,tempb,low,up)
            
    left=ncLExp(a[0],b[0],x[0])
    right=ncRExp( a[m-1], b[m-1],x[m]  )       
    ncs=np.insert(ncs,0,values=left )    
    ncs=np.append(ncs,right )
    return ncs

def weights(w):
      return w/sum(w)      


####### Random Variables 
      
#Probability density functions
def pdfExp(a,b,low,up,x): #for x in x_1< x<_ x_2 and x in x_m-1<x<x_m
    val=0
    if(low<=x and x<=up ):
        val=math.exp(b*x+a )
        return val
    else:
        return val


def pdfLExp(a1,b1,up,x): #for the x<=x_1
    c=math.exp(a1)/b1*(math.exp(b1*up )-1)
    val=0
    if( x<=up ):
        val=1/c*math.exp(b1*x+a1 )
        return val
    else:
        return val
    
def pdfRExp(am_1,bm_1,low,x): #for the x>=x_m
    c= -math.exp(am_1)/(bm_1)*(math.exp(bm_1*low ))
    val=0
    if( x>=low ):
        val=1/c*math.exp(bm_1*x+am_1 )
        return val
    else:
        return val

    
def pdfDexp(a,b,dom,x):   #for x in x_i <=x <=x_{i+1} and 2<=i<=M-2
    i=1
    k=(a[i-1]-a[i+1])/(b[i+1]-b[i-1]) #intersection b/w lines
    #normalizing constant c
    c= math.exp( a[i-1] )/ b[i-1]*( math.exp( b[i-1]*k )-math.exp( b[i-1]*dom[i] )   )   
    c=c+math.exp( a[i+1] )/ b[i+1]*( math.exp( b[i+1]*dom[i+1] )-math.exp( b[i+1]*k )   )
    val=0
    if( dom[i-1]<=x and x<=k ):
        val=1/c*math.exp( b[i-1]*x+a[i-1] )
        return val
    elif(k<=x and x<=dom[i+1]):
        val=1/c*math.exp( b[i+1]*x+a[i+1] )
        return val
    else:
        return val 
        

#Random variable simulation
    
def rvsExp(a,b,low,up,n=1):  #for x in x_1< x<_ x_2 and x in x_m-1<x<x_m
    c=math.exp(a)/b*( math.exp(b*up)-math.exp(b*low) )
    sims=np.zeros(n)
    for l in range(n):
        u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
        sims[l]=1/b*math.log( c*b*math.exp(-a)*u+math.exp(b*low) )
    return(sims)  
    
    
def rvsDexp(a,b,x,n=1): #for x in x_i <=x <=x_{i+1} and 2<=i<=M-2
    i=1
    k=(a[i-1]-a[i+1])/(b[i+1]-b[i-1]) #intersection b/w lines
    #normalizing constant c
    c= math.exp( a[i-1] )/ b[i-1]*( math.exp( b[i-1]*k )-math.exp( b[i-1]*x[i] )   )   
    c=c+math.exp( a[i+1] )/ b[i+1]*( math.exp( b[i+1]*x[i+1] )-math.exp( b[i+1]*k )   )
    #cutpoint of distribution
    cp=(1/c)*math.exp( a[i-1] )/ b[i-1]*( math.exp( b[i-1]*k )-math.exp( b[i-1]*x[i] )   )   
    sims=np.zeros(n)
    for l in range(n):
        u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
        if( u<=cp):
            sims[l]=1/b[i-1]*math.log(c*b[i-1]*math.exp(-a[i-1])*u+math.exp(b[i-1]*x[i]))
        else:
            arg=-b[i+1]/b[i-1]*math.exp(a[i-1]-a[i+1])*( math.exp(b[i-1]*k)-math.exp( b[i-1]*x[i])) 
            arg=arg+c*b[i+1]*math.exp(-a[i+1])*u+math.exp(b[i+1]*k)
            sims[l]=math.log(arg)/b[i+1]
    return(sims)


def rvsLExp(a1,b1,up,n=1 ): #for x<x_1
    c=math.exp(a1)/b1*(math.exp(b1*up )-1)
    sims=np.zeros(n)
    for l in range(n):
        u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
        sims[l]=1/b1*math.log(  c*b1*math.exp(-a1)*u+1  )
    return(sims)
    

def rvsRExp(am_1,bm_1,low,n=1): #for the x>=x_m
    c= -math.exp(am_1+bm_1*low)/(bm_1)
    sims=np.zeros(n)
    for l in range(n):
        u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
        sims[l]=(1/bm_1)*math.log(  c*bm_1*math.exp(-am_1)*u+math.exp(bm_1*low) )
    return(sims)
    
    
####### Free derivative Envelope
#

def chooseD(x,w): #returns distribution to be sampled of mixture distribution
    m=len(x)-1 #number of partitions
    parts=np.arange(m+2)
    randD = scipy.stats.rv_discrete(a=0, b=parts[m-1],name='randD', values=(parts, w))
    return randD.rvs(size=1)  


def update(x): #once you have a new x you need to update the parameters
    x=np.sort(x)
    m=len(x)-1    
    ncs=np.zeros(m) 
    a=np.zeros(m) #intercepts
    b=np.zeros(m) #slopes
    abss(a,b,x) 
    ncs=NC(x)    #Normalizing constants given x
    w=weights(ncs) #weights of mixture distribution
    parts=m+2
    return x,m,a,b,ncs,w, parts



#Derivative free ARS envelope
def DFEnvelope(x, nacc=30): #nacc is the number of accepted for the envelope
    x=np.sort(x)
    m=len(x)-1 #number of partitions within x
    ncs=np.zeros(m) #normalizing constants
    a=np.zeros(m) #intercepts
    b=np.zeros(m) #slopes
    abss(a,b,x) 
    ncs=NC(x)  #Normalizing constants given x
    w=weights(ncs) #weights of mixture distribution
    parts=m+2 #number of partitions
    j=0
    while(j<nacc):

        if(j%10==0):
            print(j)
            
        i=chooseD(x,w)[0]
           
        if(i==0): #x<x_0
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsLExp(a[0],b[0],x[0],1 )
            if(u <= f(sim)/pdfLExp(a[0],b[0],x[0],sim) ):
                x=np.append(x,sim )
                x,m,a,b,ncs,w,parts =update(x)
                j=j+1
        
        elif(i==1): #[x_0, x_1]
            low=x[0]
            up=x[1]
            tempa,tempb=ab(np.array([x[1],x[2]] ) )
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsExp(tempa,tempb,low,up,1)
            if(u <= f(sim)/pdfExp(tempa,tempb,low,up,sim)):
                x=np.append(x,sim )
                x,m,a,b,ncs,w,parts =update(x)
                j=j+1
        
        elif(i>1 and i<parts-2): #other partitions bw first and last
            tempx=x[i-2:i+1]
            tempa=a[i-2:i+1]
            tempb=b[i-2:i+1]
            sim=rvsDexp(tempa,tempb,tempx,1) 
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1) 
            if(u <= f(sim)/pdfDexp(tempa,tempb,tempx,sim)):
                x=np.append(x,sim )
                x,m,a,b,ncs,w,parts =update(x)
                j=j+1
            
        elif(i==parts-2): #[x_m-1, x_m]
            low=x[m-1]
            up=x[m]
            tempa,tempb=ab(np.array([x[m-2],x[m-1]] ) )
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsExp(tempa,tempb,low,up,1)
            
            if(u <= f(sim)/pdfExp(tempa,tempb,low,up,sim)):
                x=np.append(x,sim )
                x,m,a,b,ncs,w,parts =update(x)
                j=j+1
            
        elif(i==parts-1): #x>xm
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsRExp(a[m-1],b[m-1],x[m],1 )
            if(u <= f(sim)/pdfRExp(a[m-1],b[m-1],x[m],sim) ):
                x=np.append(x,sim )
                x,m,a,b,ncs,w,parts =update(x)
                j=j+1

    y=x.copy() #Envelope
    return y



#Once you have the envelope you can sample 
#y is the envelope


def dfARS(y,size=100): 
    
    m=len(y)-1 
    ncs=np.zeros(m) 
    a=np.zeros(m)
    b=np.zeros(m)
    abss(a,b,y) 
    ncs=NC(y)  
    w=weights(ncs)
    parts=m+2 
    sample=np.zeros(size)
    j=0
    
    while(j<size):

        if(j%25==0):
            print("Samples obtained:",j)
            
        i=chooseD(y,w)[0]

        if(i==0): 
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsLExp(a[0],b[0],y[0],1 )
            if(u <= f(sim)/pdfLExp(a[0],b[0],y[0],sim) ):
                sample[j]=sim
                j=j+1
        
        elif(i==1): 
            low=y[0]
            up=y[1]
            tempa,tempb=ab(np.array([y[1],y[2]] ) )
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsExp(tempa,tempb,low,up,1)
            if(u <= f(sim)/pdfExp(tempa,tempb,low,up,sim)):
                sample[j]=sim
                j=j+1
        
        
        elif(i>1 and i<parts-2):
            tempy=y[i-2:i+1]
            tempa=a[i-2:i+1]
            tempb=b[i-2:i+1]
            sim=rvsDexp(tempa,tempb,tempy,1) 
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1) 
            if(u <= f(sim)/pdfDexp(tempa,tempb,tempy,sim)):
                sample[j]=sim
                j=j+1
        
            
        elif(i==parts-2): 
            low=y[m-1]
            up=y[m]
            tempa,tempb=ab(np.array([y[m-2],y[m-1]] ) )
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsExp(tempa,tempb,low,up,1)
            
            if(u <= f(sim)/pdfExp(tempa,tempb,low,up,sim)):
               sample[j]=sim
               j=j+1
        
            
        elif(i==parts-1): 
            u=scipy.stats.uniform.rvs(size=1, loc=0, scale=1)
            sim=rvsRExp(a[m-1],b[m-1],y[m],1 )
            if(u <= f(sim)/pdfRExp(a[m-1],b[m-1],y[m],sim) ):
                sample[j]=sim
                j=j+1
    sample=np.append(sample,y)
    return sample

    
###Examples 


z=np.array([0.1485,0.9612,1.6783, 2.69])

#exp 1
y=DFEnvelope(z, 30)
n=len(y)-4
stats.kstest(y,'gamma',args=(2,))
name1="exp1hist"
name2="exp1qqplot"
sns.distplot(y).set_title("Envelope histogram, n="+str(n)) #density plot
plt.savefig(name1, dpi = 1000)
plt.show()
sm.qqplot(y, stats.gamma ,distargs=(2,), line='45')
plt.savefig(name2, dpi = 1000)
plt.show()
min(y)
max(y)


#exp 2
y=DFEnvelope(z, 50)
n=len(y)-4
stats.kstest(y,'gamma',args=(2,))
name1="exp2hist"
name2="exp2qqplot"
sns.distplot(y).set_title("Envelope histogram, n="+str(n)) #density plot
plt.savefig(name1, dpi = 1000)
plt.show()
sm.qqplot(y, stats.gamma ,distargs=(2,), line='45')
plt.savefig(name2, dpi = 1000)
plt.show()
min(y)
max(y)



#exp 3
y=DFEnvelope(z, 250)
n=len(y)-4
stats.kstest(y,'gamma',args=(2,))
name1="exp3hist"
name2="exp3qqplot"
sns.distplot(y).set_title("Envelope histogram, n="+str(n)) #density plot
plt.savefig(name1, dpi = 1000)
plt.show()
sm.qqplot(y, stats.gamma ,distargs=(2,), line='45')
plt.savefig(name2, dpi = 1000)
plt.show()
min(y)
max(y)


#exp 4
y=DFEnvelope(z, 500)
n=len(y)-4
stats.kstest(y,'gamma',args=(2,))
name1="exp4hist"
name2="exp4qqplot"
sns.distplot(y).set_title("Envelope histogram, n="+str(n)) #density plot
plt.savefig(name1, dpi = 1000)
plt.show()
sm.qqplot(y, stats.gamma ,distargs=(2,), line='45')
plt.savefig(name2, dpi = 1000)
plt.show()
min(y)
max(y)



sample=dfARS(y,10000)
sns.distplot(sample,bins=12, hist=True, kde=False).set_title("Envelope histogram, n=10000") #density plot
plt.savefig('sim1hist', dpi = 1000)
plt.show()
 
stats.kstest(sample,'gamma',args=(2,))
sm.qqplot(sample, stats.gamma ,distargs=(2,) , line='45')

