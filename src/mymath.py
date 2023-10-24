from Variable import Function
import numpy as np


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x *gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def add(x0, x1):
    f = Add()
    return f(x0, x1)



def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)



# A = Square()
# B = Exp()
# C = Square()

# x = Variable(np.array(2.0))
# y = add(x, x)
# y.backward()
# print(x.grad)

# y = add(add(x, x), x)
# 
# print(x.grad)
# a = square(x)
# y = add(square(a), square(a))
# y.backward()
# print(y.data)
# print(x.grad)
# b = exp(a)
# y = square(b)

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad= B.backward(b.grad)
# x.grad = A.backward(a.grad)

# y.grad = np.array(1.0)
# y.backward()
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)


# print(x.grad)