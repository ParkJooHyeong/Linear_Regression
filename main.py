import numpy as np
import matplotlib.pyplot as plt
import Gradient_Descent as gd
import Normal_Equation as ne

x = np.random.rand(1,100)
y = 4+3*x+np.random.randn(1,100)*0.3

# Gradient Descent
d = gd.model(x, y, num_iterations = 1000, learning_rate = 0.02, print_cost = True)

print("\nUsing Gradient Descent")
print ("w = " + str(d["w"]))
print ("b = " + str(d["b"]))

y_hat = np.dot(np.transpose(d["w"]),x)+d["b"]


# Normal Equation
print("\nUsing Normal Equation")
x_ne, y_hat_ne = ne.normal_equation(x,y)



plt.figure(1)
plt.plot(x, y,'b.')
plt.plot(np.transpose(x),np.transpose(y_hat), color='red', linewidth=2, label='predicted')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.figure(2)
plt.plot(x,y,'b.')
plt.plot(x_ne, y_hat_ne,'r', linewidth=2, label='predicted')
plt.title('Linear Regression using Normal Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.show()
