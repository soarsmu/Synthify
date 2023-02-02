import numpy as np
from .environment import Environment

def cooling():
  A = np.matrix([
    [1.01,0.01,0],
    [0.01,1.01,0.01],
    [0.0,0.01,1.01]])
  B = np.matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

  #intial state space
  s_min = np.array([[  1.6],[ 1.6], [1.6]])
  s_max = np.array([[  3.2],[ 3.2], [3.2]])

  Q = np.eye(3)
  R = np.eye(3)

  x_min = np.array([[-3.2],[-3.2],[-3.2]])
  x_max = np.array([[3.2],[3.2],[3.2]])
  u_min = np.array([[-1.],[-1.],[-1.]])
  u_max = np.array([[ 1.],[ 1.],[ 1.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, bad_reward=-1000)
  return env