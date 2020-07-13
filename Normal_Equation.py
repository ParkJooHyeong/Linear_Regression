import sklearn
import numpy as np
import matplotlib.pyplot as plt

X=2*np.random.rand(100,1)
Y=4+3*X+np.random.randn(100,1)


plt.plot(X,Y,'b.')
plt.title('Using normal equation')
plt.axis([0,2,0,15])
plt.show()
