import numpy as np
from .environment import Environment

def dcmotor():
  A = np.matrix([[0.98965,1.4747e-08],
    [7.4506e-09,0]
    ])

  B = np.matrix([[128],
    [0]
    ])

  #intial state space
  s_min = np.array([[-1.0],[-1.0]])
  s_max = np.array([[ 1.0],[ 1.0]])

  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-1.5],[-1.5]])
  x_max = np.array([[ 1.5],[ 1.5]])
  u_min = np.array([[-1.]])
  u_max = np.array([[ 1.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  return env