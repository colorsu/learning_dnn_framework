from Function import Function
from Variable import Variable
import numpy as np


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# A = Square()
# B = Exp()
# C = Square()

# x = Variable(np.array(0.5))
# a = A(x)
# b = B(a)
# y = C(b)

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad= B.backward(b.grad)
# x.grad = A.backward(a.grad)

# print(x.grad)