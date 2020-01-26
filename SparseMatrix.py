import numpy as np


class SparseMatrix:
    """
    Description
    -----------
    Class for sparse matrix
    """
    def __init__(self, shape=None, zeros=True):
        self.A = {}
        if shape is not None:
            self.shape = shape
        self.zeros = zeros

    def get(self, index):
        if index in self.A:
            return self.A[index]
        else:
            return 0

    def set(self, index, val):
        self.A[index] = val

    def add(self, index, val):
        if index in self.A:
            self.A[index] += val
        else:
            self.A[index] = val

    def sub(self, index, val):
        if index in self.A:
            self.A[index] -= val
        else:
            self.A[index] = -val

    def dot(self, b):
        """
        Dot product of n*n symmetric sparse matrix and n-vector
        Parameters
        ----------
        b: 1d numpy.array
        Returns
        -------
        1d numpy.array x such that Ab = x
        """
        result = np.zeros(len(b))
        for row, col in self.A:
            if row > col:
                result[row] += self.A[(row, col)] * b[col]
                result[col] += self.A[(row, col)] * b[row]

        result += [self.A[(i, i)] * b[i] for i in range(len(b))]
        return result