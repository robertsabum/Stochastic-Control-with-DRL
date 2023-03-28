import numpy as np


# nxn matrix of random tuples containing mean and standard deviation
mcs = np.random.uniform(0, 1, size=(5, 5))
mct = np.random.uniform(0, 1, size=(5, 5))

# stack the two matrices
mcu = np.stack((mcs, mct), axis=2)
print(np.eye(5))

def foo(parameters):
    return np.random.normal(parameters[0], parameters[1])

# vectorize the function
vfoo = np.vectorize(foo, signature='(n)->()')

# apply the function to the matrix
print(vfoo((1, 2)))

