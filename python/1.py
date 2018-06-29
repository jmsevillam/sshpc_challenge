import numpy as np
import matplotlib.pylab as plt
Arch=np.genfromtxt('datos.txt')
plt.hist(Arch,bins=100,color='darkorange',alpha=0.9)
plt.xlabel(r'$R^2$')
plt.ylabel('Frequencies')
plt.title('Distribution of $R^2$')
plt.savefig('Rs.pdf')
# plt.show()
plt.close()
# a=0
# for i in range(len(Arch)):
#     if Arch[i]**2 > 0.8:
#         a +=1
# print a
