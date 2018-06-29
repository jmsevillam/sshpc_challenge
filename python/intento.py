import numpy as np
import random
import sys
import math
from scipy.stats import f
from scipy.stats import norm
def CalculoNOVA(X,Y,n,p):
    XT=np.transpose(X)
    YT=np.transpose(Y)
    Inv=np.linalg.inv(np.dot(XT,X))
    Hhat=np.dot(X,np.dot(Inv,XT))
    SST=np.dot(YT,np.dot((np.identity(n)-1.0/n*np.ones((n,n))),Y))
    SSR=np.dot(YT,np.dot((Hhat-1.0/n*np.ones((n,n))),Y))
    SSE=SST-SSR
    Rsq=SSR[0,0]/SST[0,0]
    F=(Rsq*(n-p-1))/((1-Rsq)*p)
    if F>f.ppf(0.9,p,n-(p+1)):
        sigmasq=SSE[0,0]/(n-1)
        beta=np.dot(np.dot(Inv,XT),Y)
        Pi=np.array([0]*p)
        for i in range(p):
            t0=beta[i]/math.sqrt(sigmasq*Inv[i,i])
            Pvalue=2*(1-norm.cdf(t0))
            if Pvalue<0.25:
                Pi[i]=1
                
            else:
                Pi[i]=0
        pprime=sum(Pi)
        aux=0
        Xnew=np.zeros((n,pprime))
        for i in range(p):
            if Pi[i] == 1:
                Xnew[:,aux]=X[:,i]
                aux+=1
        return Rsq,F,Xnew,pprime
    else:
        quit()
seed=int(sys.argv[1])
Size=int(sys.argv[2])
random.seed(seed)
np.random.seed(seed)
nfila=4*Size # numero de mediciones
pcol=Size # numero de variables
Xdatos=np.random.normal(0,1,(nfila,pcol))
Ydatos=np.random.normal(0,1,(nfila,1))
Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xdatos,Ydatos,nfila,pcol)
Rs2, Fs2, Xnuevos, pprima2 = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
print Rs1, Fs1, Rs2, Fs2, pprima




# ----------------
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
# Rs1, Fs1, Xnuevos, pprima = CalculoNOVA(Xnuevos,Ydatos,nfila,pprima)
# print Rs1, Fs1,pprima
