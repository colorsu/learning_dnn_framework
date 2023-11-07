if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero.utils import plot_dot_graph
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1,2,3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

a = Variable(np.array([[1,2,3], [4, 5, 6]]))
z = F.sum(a, axis=0)
z.backward()
print(z)
print(a.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)