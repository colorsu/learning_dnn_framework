from src.Function import Function
from src.Variable import Variable
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


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


# A = Square()
# B = Exp()
# C = Square()

x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad= B.backward(b.grad)
# x.grad = A.backward(a.grad)

y.grad = np.array(1.0)
y.backward()
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)


print(x.grad)