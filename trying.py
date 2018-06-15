import numpy as np
import random
import sys
from scipy.stats import f
from scipy.stats import norm
seed=int(sys.argv[1])
Size=int(sys.argv[2])
random.seed(seed)
np.random.seed(seed)
n=4*Size # numero de mediciones
p=Size # numero de variables
mu=0.0 
sigma=1
#Creamos los arreglos X y Y, los cuales se llenan de forma aleatoria
X=[[random.gauss(mu,sigma) for i in range(p)] for j in range(n) ]
Y=[[random.gauss(mu,sigma) for j in range(1)] for i in range(n) ]
X=np.array(X)
Y=np.array(Y)
#se calculan sus respectivas transpuestas
XT=np.transpose(X)
YT=np.transpose(Y)
# calculamos la inversa de la matriz del producto entre X y su traspuesta
Inv=np.linalg.inv(np.dot(XT,X))
beta1=np.dot(Inv,XT)
# Se calculan los beta del ajuste
beta=np.dot(beta1,Y)
# # -----------matriz hat para el ajuste-------------
Hhat=np.dot(X,beta1)
# # ---------------los y's para sacar la regresion------------
Yideal=np.dot(X,beta)
# -----------------------------------
# #determinamos SST
SST1=np.dot((np.identity(n)-1.0/n*np.ones((n,n))),Y)
SST=np.dot(YT,SST1)
# Determinamos SSR
SSR1=np.dot((Hhat-1.0/n*np.ones((n,n))),Y)
SSR=np.dot(YT,SSR1)
# Determinamos SSE
SSE1=np.dot((np.identity(n)-Hhat),Y)
SSE=np.dot(YT,SSE1)
# Se Calcula R2
Rsq1=SSR[0,0]/SST[0,0]
Rsq2=1-(SSE[0,0])/(SST[0,0])
# Calculamos sigma2
sigma2=SSE[0,0]/(n-1)
# print sigma2, Rsq1, Rsq2
# # print sigma2
sigmamatrix=sigma2*Inv
aux=0.0
# ----------- Calculamos sigmas----------
sigmai=np.array([0.0]*Size)
for i in range(Size):
    sigmai[i]=sigmamatrix[i,i]
sigmai=np.sqrt(sigmai)
MSE=SSE[0,0]/(n-p-1)
# Calculamos el MSR
MSR=SSR[0,0]/p
# Calculamos el MST
MST=SST[0,0]/(n-1)
# print 'MSR='+str(MSR),'MST='+str(MST),'MSE='+str(MSE) 
# ------------Calculamos F-----------
F=(Rsq1*(n-p-1))/((1-Rsq1)*p)
# Rango1=0.75
# Rango2=0.95

Rango=0.9
Ftest=f.ppf(Rango,p,n-(p+1))
Pi=np.array([0]*p)
if F>Ftest:
    tzeros=beta[:,0]/sigmai
    Pvalue=2*(1-norm.cdf(tzeros))
    for i in range(p):
        if Pvalue[i]<0.5:
            Pi[i]=1
        else:
            Pi[i]=0
else:
    quit()
pprime=sum(Pi)
Xnew=np.zeros((n,pprime))
aux=0
for i in range(p):
    # print i
    if Pi[i]==1:
        # print X[:,i]
        Xnew[:,aux]=X[:,i]
        aux+=1
# ------------- Segunda pasada --------------------
XnewT=np.transpose(Xnew) 
Invnew=np.linalg.inv(np.dot(XnewT,Xnew))
beta1new=np.dot(Invnew,XnewT)
# Se calculan los beta del ajuste new
betanew=np.dot(beta1new,Y)
Hhatnew=np.dot(Xnew,beta1new)
# # ---------------los y's para sacar la regresion new------------
Yidealnew=np.dot(Xnew,betanew)
# -----------------------------------
# # El SST es igual para todos
# Determinamos SSRnew
SSR1new=np.dot((Hhatnew-1.0/n*np.ones((n,n))),Y)
SSRnew=np.dot(YT,SSR1new)
# Determinamos SSEnew
SSE1new=np.dot((np.identity(n)-Hhatnew),Y)
SSEnew=np.dot(YT,SSE1new)
# Se Calcula R2
Rsq1new=SSRnew[0,0]/SST[0,0]
# Rsq2=1-(SSE[0,0])/(SST[0,0])
# Calculamos sigma2
sigma2new=SSEnew[0,0]/(n-1)
# print sigma2, Rsq1, Rsq2
# # print sigma2
sigmamatrixnew=sigma2new*Invnew
# ----------- Calculamos sigmas----------
sigmainew=np.array([0.0]*pprime)
for i in range(pprime):
    sigmainew[i]=sigmamatrixnew[i,i]
sigmainew=np.sqrt(sigmainew)
MSEnew=SSEnew[0,0]/(n-pprime-1)
# Calculamos el MSR
MSRnew=SSRnew[0,0]/pprime

# print 'MSR='+str(MSR),'MST='+str(MST),'MSE='+str(MSE) 
# ------------Calculamos F-----------
Fnew=(Rsq1new*(n-pprime-1))/((1-Rsq1new)*pprime)
# Rango1=0.75
# Rango2=0.95

Rango=0.9
Ftestnew=f.ppf(Rango,pprime,n-(pprime+1))
Pinew=np.array([0]*pprime)
if Fnew>Ftestnew:
    tzerosnew=betanew[:,0]/sigmainew
    Pvaluenew=2*(1-norm.cdf(tzerosnew))
    for i in range(pprime):
        if Pvaluenew[i]<0.5:
            Pi[i]=1
        else:
            Pi[i]=0
print Rsq1new, Fnew
