if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero.utils import plot_dot_graph
from dezero import Variable
import dezero.functions as F

x0 = Variable(np.array([1, 2, 3]))
# x0 = Variable(np.array([9]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)