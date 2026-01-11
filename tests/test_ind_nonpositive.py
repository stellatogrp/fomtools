import numpy as np
from algoverify.proxable_functions import IndNonPositive

print("Testing IndNonPositive implementation")
print()

# Create indicator function
indicator = IndNonPositive()

# Test 1: Point with all non-positive components
print("Test 1: Point with all non-positive components")
x_nonpos = np.array([-1.0, -2.0, -0.5, -3.0])
print(f"x = {x_nonpos}")
print(f"All non-positive: {np.all(x_nonpos <= 0)}")
print(f"eval(x) = {indicator.eval(x_nonpos)}")
prox_value = indicator.prox(x_nonpos.copy(), gamma=1.0)
y_prox = np.minimum(x_nonpos, 0)
print(f"prox() returned (indicator value): {prox_value}")
print(f"Projected point should be x: {np.allclose(x_nonpos, y_prox)}")
print(f"eval(projected point) = {indicator.eval(y_prox)}")
print()

# Test 2: Point with all positive components
print("Test 2: Point with all positive components")
x_pos = np.array([1.0, 2.0, 0.5, 3.0])
print(f"x = {x_pos}")
print(f"All non-positive: {np.all(x_pos <= 0)}")
print(f"eval(x) = {indicator.eval(x_pos)}")
prox_value_pos = indicator.prox(x_pos.copy(), gamma=1.0)
y_prox_pos = np.minimum(x_pos, 0)
print(f"prox() returned (indicator value): {prox_value_pos}")
print(f"Projected point = {y_prox_pos}")
print(f"All projected components <= 0: {np.all(y_prox_pos <= 0)}")
print(f"eval(projected point) = {indicator.eval(y_prox_pos)}")
print()

# Test 3: Point with mixed signs
print("Test 3: Point with mixed signs")
x_mixed = np.array([-1.0, 2.0, -0.5, 3.0, -2.0])
print(f"x = {x_mixed}")
print(f"All non-positive: {np.all(x_mixed <= 0)}")
print(f"eval(x) = {indicator.eval(x_mixed)}")
prox_value_mixed = indicator.prox(x_mixed.copy(), gamma=1.0)
y_prox_mixed = np.minimum(x_mixed, 0)
print(f"prox() returned (indicator value): {prox_value_mixed}")
print(f"Projected point = {y_prox_mixed}")
print(f"All projected components <= 0: {np.all(y_prox_mixed <= 0)}")
print(f"Projection formula correct: {np.allclose(y_prox_mixed, np.minimum(x_mixed, 0))}")
print(f"eval(projected point) = {indicator.eval(y_prox_mixed)}")
print()

# Test 4: Point with zeros
print("Test 4: Point with zeros")
x_zero = np.array([0.0, -1.0, 0.0, 2.0])
print(f"x = {x_zero}")
print(f"All non-positive: {np.all(x_zero <= 0)}")
print(f"eval(x) = {indicator.eval(x_zero)}")
prox_value_zero = indicator.prox(x_zero.copy(), gamma=1.0)
y_prox_zero = np.minimum(x_zero, 0)
print(f"prox() returned (indicator value): {prox_value_zero}")
print(f"Projected point = {y_prox_zero}")
print(f"eval(projected point) = {indicator.eval(y_prox_zero)}")
print()

# Test 5: Multiple random points
print("Test 5: Testing multiple random points")
np.random.seed(42)
for i in range(5):
    x_test = np.random.normal(size=4)
    prox_val = indicator.prox(x_test.copy(), gamma=1.0)
    y_prox_test = np.minimum(x_test, 0)
    all_nonpos = np.all(y_prox_test <= 0)
    print(f"  Point {i+1}: has positive = {np.any(x_test > 0)}, "
          f"all prox <= 0: {all_nonpos}, prox() value: {prox_val}")

print("\nAll tests completed!")

