import numpy as np

from .environment import PolySysEnvironment

# Show that there is an invariant that can prove the policy safe
def self_driving():
  # 2-dimension and 1-input system
  ds = 2
  us = 1

  #the speed is set to 2 in this case
  v = 2
  cl = 2
  cr = -2

  def f(x, u):
    delta = np.zeros((ds, 1), float)
    delta[0, 0] = -v*(x[1,0] - ((pow(x[1,0],3))/6))
    delta[1, 0] = u[0, 0]                   #angular velocity (controlled by AIs)
    return delta

  def K_to_str (K):
    #Control policy K to text
    nvars = len(K[0])
    X = []
    for i in range(nvars):
      X.append("x[" + str(i+1) + "]")

    ks = []
    for i in range(len(K)):
      strstr = ""
      for k in range(len(X)):
        if (strstr is ""):
          strstr = str(K[i,k]) + "*" + X[k]
        else:
          strstr = strstr + "+" + str(K[i,k]) + "*" + X[k]
      ks.append(strstr)
    return ks

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("-{}*(x[2] - ((x[2]^3)/6))".format(v))
    f.append(kstr[0])
    return f

  h = 0.1

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  pi = 3.1415926

  #intial state space
  s_min = np.array([[-1],[-pi/4]])
  s_max = np.array([[ 1],[ pi/4]])

  u_min = np.array([[-10]])
  u_max = np.array([[10]])

  #the only portion of the entire state space that our verification is interested.
  bound_x_min = np.array([[None],[-pi/2]])
  bound_x_max = np.array([[None],[ pi/2]])

  #reward functions
  Q = np.zeros((2,2), float)
  np.fill_diagonal(Q, 1)
  R = np.zeros((1,1), float)
  np.fill_diagonal(R, 1)

  #user defined unsafety condition
  def unsafe_eval(x):
    outbound1 = -(x[0,0]- cr)*(cl-x[0,0])
    if (outbound1 >= 0):
      return True
    return False
  def unsafe_string():
    return ["-(x[1]- {})*({}-x[1])".format(cr, cl)]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x)) -np.dot(u.T,R.dot(u))
    if (unsafe_eval(x)):
      reward -= 100
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      return -1
    return 0

  env = PolySysEnvironment(f, f_to_str, rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, bound_x_min=bound_x_min, bound_x_max=bound_x_max, timestep=0.1)
  return env