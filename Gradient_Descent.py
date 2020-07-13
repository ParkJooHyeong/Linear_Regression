import numpy as np
import matplotlib.pyplot as plt
import functions as func

x = np.random.rand(1,100)
y = 4+3*x+np.random.randn(1,100)

# d = func.model(x, y, num_iterations = 1000, learning_rate = 0.02, print_cost = True)

# print ("w = " + str(d["w"]))
# print ("b = " + str(d["b"]))
#
# y_hat = np.dot(np.transpose(d["w"]),x)+d["b"]
plt.scatter(x, y,label='data')
# plt.plot(np.transpose(x),np.transpose(y_hat), color='red', linewidth=2, label='predicted')
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
