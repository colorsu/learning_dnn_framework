if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero.utils import plot_dot_graph
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)
y = y.reshape((2, 3))
print(y)

y = F.transpose(y)
y.cleargrad()
x.cleargrad()
y.backward()

print(x.grad)
print(y)