import numpy as np
import matplotlib.pylab as plt
Arch=np.genfromtxt('tabla.txt')
fig = plt.figure(figsize=(10,7))
plt.subplots_adjust(wspace=0.2,hspace=0.4)
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
ax1.hist(Arch[:,0],bins=100,color='darkorange',alpha=0.7)
ax1.set_xlabel(r'$R^2$')
ax1.set_ylabel('Frequencies')
ax1.set_title('Distribution of $R^2$')
ax2.hist(Arch[:,1],bins=100,color='royalblue',alpha=0.7)
ax2.set_xlabel(r'$F$')
ax2.set_ylabel('Frequencies')
ax2.set_title('Distribution of $F$')
ax3.hist(Arch[:,2],bins=100,color='darkorange',alpha=0.7)
ax3.set_xlabel(r'$R^2$')
ax3.set_ylabel('Frequencies')
ax3.set_title('Distribution of $R^2$')
ax4.hist(Arch[:,3],bins=100,color='royalblue',alpha=0.7)
ax4.set_xlabel(r'$F$')
ax4.set_ylabel('Frequencies')
ax4.set_title('Distribution of $F$')
plt.savefig('RandF025.pdf')
plt.show()
# plt.close()
# a=0
# for i in range(len(Arch)):
#     if Arch[i]**2 > 0.8:
#         a +=1
# print a
