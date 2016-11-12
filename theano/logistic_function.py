import numpy as np
import theano.tensor as T
from theano import function, In


def main():
    x = T.dmatrix('x')

    # T.exp
    s = 1 / (1 + T.exp(-x))
    logistic = function([x], s)

    # 0 is 0.5, negative < 0.5...
    print(logistic([[0, 1], [-1, -2]]))
    # logistic function can be expressed with hyperbolic tan term
    s2 = (1 + T.tanh(x / 2)) / 2
    logistic2 = function([x], s2)
    print(np.allclose(logistic([[0, 1], [-1, -2]]),
                      logistic2([[0, 1], [-1, -2]])))

    # do more things at a time
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_squared = diff**2
    f = function([a, b], [diff, abs_diff, diff_squared])
    print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))

    # default value
    x, y = T.dscalars('x', 'y')
    z = x + y
    f = function([x, In(y, value=1)], z)
    print(f(33))
    print(f(33, 2))

    # Inputs with default values must follow inputs without default
    # values (like Python’s functions). There can be multiple inputs
    # with default values. These parameters can be set positionally
    # or by name, as in standard Python
    x, y, w = T.dscalars('x', 'y', 'w')
    z = (x + y) * w
    f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
    print(f(33))
    print(f(33, 2))
    print(f(33, 0, 1))
    print(f(33, w_by_name=1))
    print(f(33, w_by_name=1, y=0))


if __name__ == '__main__':
    main()