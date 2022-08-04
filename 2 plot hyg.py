import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -1/x**2

def f1(x):
    return -1/(x**2+eps)

eps=0.0003

x=np.linspace(0,1,100)


plt.title("Plot of $-1/x$")
plt.xlabel('x')
plt.ylabel(r'f(x)')

plt.plot(x,f(x))
plt.plot(x,f1(x))


legend1=plt.legend(['without eps','with eps'], loc =4)
ax = plt.gca().add_artist(legend1)
plt.tight_layout()
plt.show()