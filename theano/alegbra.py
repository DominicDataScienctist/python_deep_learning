import numpy as np
import theano.tensor as T
from theano import function


def main():

    # define two instance of dscalar, x, y are variables
    x = T.dscalar('x')
    y = T.dscalar('y')

    # combine x, y into another variable
    z = x + y

    # create a function
    # The first argument to function is a list of Variables that will be
    # provided as inputs to the function. The second argument is a single
    # Variable or a list of Variables. For either case, the second argument
    # is what we want to see as output when we apply the function. f may
    # then be used like a normal Python function.
    f = function([x, y], z)
    print(f(2, 3))
    print(np.allclose(f(16.3, 12.1), 28.4))

    # as a shortcut, can also use eval method of a teano object instead of
    # definding a function
    print(np.allclose(z.eval({x: 16.3, y: 12.1}), 28.4))


    # Matrix
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = function([x, y], z)

    # it output numpy array when given lists of lists as inputs
    print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

    # of coz it can use numpy as input as well
    print(f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])))

    # vector excerise
    a = T.vector() # declare variable
    b = T.vector()
    out = a ** 2 + b ** 2 + 2 * a * b            # build symbolic expression
    f = function([a, b], out)   # compile function
    print(f([0, 1, 2], [3, 4, 5]))

if __name__ == '__main__':
    main()
