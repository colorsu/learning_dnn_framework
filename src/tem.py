from Variable import Variable
from mymath import *
from tools import *
from Config import *

# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(as_array(x.data - eps))
#     x1 = Variable(as_array(x.data + eps))
#     y0 = f(x0)
#     y1 = f(x1)

#     return (y1.data - y0.data) / (2 * eps)

# f = Square()
# x = Variable(np.array(2.0))
# x = Variable(None)

# dy = numerical_diff(f, x)
# print(dy)


# Config.enable_backprop = True
# x = Variable(np.ones((100, 100, 100)))
# y = square(square(square(x)))
# y.backward()

# Config.enable_backprop = False
# x = Variable(np.ones((100, 100, 100)))
# y = square(square(square(x)))
# print(len(x))
# print(f"shape={x.shape}")
# print(x)


# with using_config('enable_backprop', False):
#     x = Variable(np.ones((100, 100, 100)))
#     y = square(square(square(x)))

# def no_grad():
#     return using_config('enable_backprop', False)

# with no_grad():
#     x = Variable(np.ones((100, 100, 100)))
#     y = square(square(square(x)))

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
y = a * b
y
