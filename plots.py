import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'

data=np.genfromtxt("data.dat")

r2b=data[::2,1]
Fb=data[::2,4]
print(r2b)
r2a=data[1::2,1]
Fa=data[1::2,4]

fig=plt.figure()
plt.subplots_adjust(left=0.11,right=0.99, bottom=0.13,top=0.97,wspace=0.16,hspace=0.25)
ax=fig.add_subplot(221)
ax.hist(r2b,bins=50)
ax.set_xlabel(r'$r^2$')
ax.set_xlim(xmin=0)
plt.ylabel('Frequency')

ax2=fig.add_subplot(222)
ax2.hist(Fb,bins=np.linspace(0,30,100),color='C1')
ax2.set_xlabel(r'$F$')
ax2.set_xlim(xmin=0)
ax3=fig.add_subplot(223)
ax3.hist(r2a,bins=50)
ax3.set_xlabel(r'$r^2$')
ax3.set_ylabel('Frequency')
ax3.set_xlim(xmin=0)
ax4=fig.add_subplot(224)
ax4.hist(Fa,bins=np.linspace(0,30,100),color='C1')
ax4.set_xlabel(r'$F$')
ax4.set_xlim(xmin=0)
plt.show()


