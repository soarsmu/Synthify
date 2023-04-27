from z3 import *

# Define the model as a function of its inputs
x = Real('x')
y = 2 * x + 1

# Define the linear function as a function of the model inputs and coefficients
a = Real('a')
b = Real('b')
f = a * x + b

# Define the objective function as the difference between the model and the linear function
obj = (y - f) ** 2

cost = Real('cost')

# Define the input range for the model
range_constraint = And(x >= 0, x <= 10)

# Create solver and add objective function and range constraint
solver = Optimize()
solver.add(range_constraint)
solver.add(cost == obj)
solver.minimize(cost)

# Check if solver is satisfiable
if solver.check() == sat:
    # Extract values of coefficients from solution
    model = solver.model()
    a_val = model.eval(a).as_decimal(10)
    b_val = model.eval(b).as_decimal(10)

    print(f"a = {a_val}, b = {b_val}")
else:
    print("No solution found.")
