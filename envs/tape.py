import numpy as np

from .environment import Environment

def tape ():
  A = np.matrix([[5.5197e-17,-3.5503e-17,6.2468e-32],
    [2.7756e-17,0,0],
    [0,2.7756e-17,0]
    ])

  B = np.matrix([[0.25],
    [0],
    [0]
    ])

  #intial state space
  s_min = np.array([[-1.0],[-1.0], [-1.0]])
  s_max = np.array([[ 1.0],[ 1.0], [ 1.0]])

  Q = np.matrix("1 0 0 ; 0 1 0; 0 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-3],[-3],[-3]])
  x_max = np.array([[ 3],[ 3], [3]])
  u_min = np.array([[-10.]])
  u_max = np.array([[ 10.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  return env