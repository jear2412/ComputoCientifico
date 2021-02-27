# Cómputo-Científico

Programas en Python de Computo Científico. Se incluirán algoritmos de metodos numéricos y de simulación estocástica; así como experimentos. 


*Descripción breve*

*1)bsfs.py*

Incluye codigo de backward y forward substitution para resolver los sistemas
   Tb=y en donde T es triangular superior
   Lb=y en donde L es triangular inferior
   
*2)GEPPLU.py*   

Incluye codigo para poder encontrar la factorizacion PALU de una matriz A, es decir PA=LU, por medio de eliminacion gaussiana con pivoteo parcial. P representa la matriz de pivoteo, U la matriz triangular superior y L la matriz triangular inferior con elementos de la diagonal igual 1. Teniendo el sistema de ecuaciones Ax=b hacemos
        PAx=Pb
        LUx=Pb
Luego se aplica backward y forward substitution. Note que el algoritmo graba U y L sobre A.  
  
*3)Cholesky.py*   
  
Dada una matriz A simetrica definida positiva (i.e x'Ax>0 para todo x) encuentra la factorizacion de Cholesky de A. Es decir A=LL^T en donde L es una matriz triangular inferior. Podemos resolver el sistema Ax=b por medio de
  Ax=b
  LL^t=b
Luego se aplica backward y forward substitution.  

El algoritmo guarda solo la parte L^T sobre la matriz A.


*4) QR.py*

Encuentra la factorizacion QR incompleta de una matriz A de orden mxn. Q es una matriz de vectores ortogonales y R es una matriz triangular superior. Suponemos que A es de rango completo. Notese que guarda la matriz Q sobre A, pues son de la misma dimension. La funcion retorna R.


*5)Gershgorin.py*

Funcion que dibuja los circulos de Gershgorin que estiman el valor de los eigenvalores de una matriz A. Ver Biswa Natha (2013)

  
*6)ARSgit.py*

Se incluye documentacion y codigo de Derivative Free Adaptive Rejection Sampling. Gilks 1992
Derivative free adaptive rejection sampling. Se incluye documentacion y ejemplos dentro del codigo. 

   
*7) t-walk *

Se incluye el muestreador MCMC t-walk por Christen y Fox (2010). Ver https://www.cimat.mx/~jac/twalk/


Christen, J. A. and Fox, C. (2010). “A general purpose sampling algorithm for continuous distributions (the t-walk).”Bayesian Analisis, 4(2): 263–282.
   
