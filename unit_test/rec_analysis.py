import numpy as np
import cvxpy as cp

p = 1 / 3
p_table = [p, 1 - p]

eta = cp.Variable(2)

signaling_table = [eta, 1 - eta]

if __name__ == '__main__':
    print('haha')
