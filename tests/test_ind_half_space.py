import numpy as np
from algoverify.proxable_functions import IndHalfSpace

# Test parameters
np.random.seed(42)
n = 5
a = np.random.normal(size=n)
norm_a_sq = np.dot(a, a)
b = 2.0

print("Testing IndHalfSpace implementation")
print(f"Half-space: a^T x <= {b}")
print(f"a = {a}")
print(f"||a||^2 = {norm_a_sq}")
print()

# Create indicator function
indicator = IndHalfSpace(a=a, b=b, norm_a=norm_a_sq)

# Helper function to compute projection manually
def compute_projection(x, a, b, norm_a_sq):
    a_dot_x = np.dot(a, x)
    if a_dot_x <= b:
        return x.copy()
    else:
        return x - ((a_dot_x - b) / norm_a_sq) * a

# Test 1: Point inside half-space
print("Test 1: Point inside half-space")
x_inside = np.zeros(n)  # a^T * 0 = 0 <= b
print(f"x = {x_inside}")
print(f"a^T x = {np.dot(a, x_inside)}")
print(f"eval(x) = {indicator.eval(x_inside)}")
prox_value = indicator.prox(x_inside.copy(), gamma=1.0)
y_prox = compute_projection(x_inside, a, b, norm_a_sq)
print(f"prox() returned (indicator value): {prox_value}")
print(f"Projected point should be x: {np.allclose(x_inside, y_prox)}")
print(f"eval(projected point) = {indicator.eval(y_prox)}")
print()

# Test 2: Point outside half-space
print("Test 2: Point outside half-space")
x_outside = 3.0 * a / np.linalg.norm(a)  # Point in direction of a, scaled
a_dot_x_outside = np.dot(a, x_outside)
print(f"x = {x_outside}")
print(f"a^T x = {a_dot_x_outside} (should be > {b})")
print(f"eval(x) = {indicator.eval(x_outside)}")
prox_value_outside = indicator.prox(x_outside.copy(), gamma=1.0)
y_prox_outside = compute_projection(x_outside, a, b, norm_a_sq)
print(f"prox() returned (indicator value): {prox_value_outside}")
print(f"Projected point = {y_prox_outside}")
print(f"a^T prox(x) = {np.dot(a, y_prox_outside)} (should be <= {b})")
print(f"eval(projected point) = {indicator.eval(y_prox_outside)}")
# Verify projection formula: y = x - ((a^T x - b) / ||a||^2) * a
expected_y = x_outside - ((a_dot_x_outside - b) / norm_a_sq) * a
print(f"Projection formula correct: {np.allclose(y_prox_outside, expected_y)}")
print()

# Test 3: Point exactly on boundary
print("Test 3: Point exactly on boundary")
x_boundary = (b / norm_a_sq) * a  # a^T x = b
print(f"x = {x_boundary}")
print(f"a^T x = {np.dot(a, x_boundary)} (should be = {b})")
print(f"eval(x) = {indicator.eval(x_boundary)}")
prox_value_boundary = indicator.prox(x_boundary.copy(), gamma=1.0)
y_prox_boundary = compute_projection(x_boundary, a, b, norm_a_sq)
print(f"prox() returned (indicator value): {prox_value_boundary}")
print(f"Projected point should be x: {np.allclose(x_boundary, y_prox_boundary)}")
print()

# Test 4: Validation - zero vector
print("Test 4: Validation - zero vector")
try:
    bad_indicator = IndHalfSpace(a=np.zeros(n), b=0.0, norm_a=0.0)
    print("ERROR: Should have raised ValueError for zero vector")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test 5: Validation - norm_a mismatch
print("Test 5: Validation - norm_a mismatch")
try:
    bad_indicator = IndHalfSpace(a=a, b=b, norm_a=norm_a_sq + 1.0)
    print("ERROR: Should have raised ValueError for norm_a mismatch")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test 6: Multiple random points
print("Test 6: Testing multiple random points")
for i in range(5):
    x_test = np.random.normal(size=n)
    prox_val = indicator.prox(x_test.copy(), gamma=1.0)
    y_prox_test = compute_projection(x_test, a, b, norm_a_sq)
    a_dot_y = np.dot(a, y_prox_test)
    print(f"  Point {i+1}: a^T x = {np.dot(a, x_test):.4f}, "
          f"a^T prox(x) = {a_dot_y:.4f}, "
          f"in half-space: {a_dot_y <= b + 1e-10}, "
          f"prox() value: {prox_val}")

print("\nAll tests completed!")

