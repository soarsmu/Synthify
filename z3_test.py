from z3 import *

w1, b1, w2, b2, w3, b3 = model.get_weights() # unpack weights from model

def Relu(x):
    return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)
def Abs(x):
    return If(x <= 0, -x, x)
def net(x):
    x1 = w1.T @ x + b1
    y1 = Relu(x1)
    x2 = w2.T @ y1 + b2
    y2 = Relu(x2)
    x3 = w3.T @ y2 + b3
    return x3

x = np.array([Real('x')])
y_true = cheb(x)
y_pred = net(x)
s = Solver()
s.add(-1 <= x[0], x[0] <= 1)
s.add(Abs( y_pred[0] - y_true[0] ) >= 0.5)
#prove(Implies( And(-1 <= x[0], x[0] <= 1),  Abs( y_pred[0] - y_true[0] ) >= 0.2))
res = s.check()
print(res)
if res == sat:
    m = s.model()
    print("Bad x value:", m[x[0]])
    x_bad = m[x[0]].numerator_as_long() / m[x[0]].denominator_as_long() 
    print("Error of prediction: ", abs(model.predict(np.array([x_bad])) - cheb(x_bad)))


x, y = Ints('x y')
F = And(x >= 1, x == 2*y)
G = And(2*y - x == 0, x >= 0)
s = Solver()
s.add(Not(F == G))
r = s.check()
if r == unsat:
    print("proved")
else:
    print("counterexample")
    print(s.model())