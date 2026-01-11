import numpy as np
from algoverify.proxable_functions import IndPoint

print("Testing IndPoint implementation")
print()

# Test parameters
np.random.seed(42)
n = 5
p = np.random.normal(size=n)

print(f"Target point p = {p}")
print()

# Create indicator function
indicator = IndPoint(p=p)

# Test 1: Point exactly equal to p
print("Test 1: Point exactly equal to p")
x_equal = p.copy()
print(f"x = {x_equal}")
print(f"x == p: {np.allclose(x_equal, p)}")
print(f"eval(x) = {indicator.eval(x_equal)}")
prox_value = indicator.prox(x_equal.copy(), gamma=1.0)
print(f"prox() returned (indicator value): {prox_value}")
print(f"Projected point should be p: {np.allclose(p, p)}")
print(f"eval(projected point) = {indicator.eval(p)}")
print()

# Test 2: Point different from p
print("Test 2: Point different from p")
x_diff = np.random.normal(size=n)
print(f"x = {x_diff}")
print(f"x == p: {np.allclose(x_diff, p)}")
print(f"eval(x) = {indicator.eval(x_diff)}")
prox_value_diff = indicator.prox(x_diff.copy(), gamma=1.0)
print(f"prox() returned (indicator value): {prox_value_diff}")
print(f"Projected point should always be p: {np.allclose(p, p)}")
print(f"eval(projected point) = {indicator.eval(p)}")
print()

# Test 3: Point very close to p (within tolerance)
print("Test 3: Point very close to p (within tolerance)")
x_close = p + 1e-12 * np.random.normal(size=n)
print(f"x = {x_close}")
print(f"x == p (within tolerance): {np.allclose(x_close, p)}")
print(f"eval(x) = {indicator.eval(x_close)}")
prox_value_close = indicator.prox(x_close.copy(), gamma=1.0)
print(f"prox() returned (indicator value): {prox_value_close}")
print()

# Test 4: Scalar point
print("Test 4: Scalar point (1D)")
p_scalar = 3.14
indicator_scalar = IndPoint(p=p_scalar)
x_scalar = 2.71
print(f"p = {p_scalar}")
print(f"x = {x_scalar}")
print(f"eval(x) = {indicator_scalar.eval(np.array([x_scalar]))}")
prox_val_scalar = indicator_scalar.prox(np.array([x_scalar]), gamma=1.0)
print(f"prox() returned (indicator value): {prox_val_scalar}")
print()

# Test 5: Multiple random points
print("Test 5: Testing multiple random points")
for i in range(5):
    x_test = np.random.normal(size=n)
    prox_val = indicator.prox(x_test.copy(), gamma=1.0)
    print(f"  Point {i+1}: prox() value: {prox_val} (should always be 0.0)")

print("\nAll tests completed!")

