import numpy as np
from algoverify.proxable_functions import IndBallLinf

print("Testing IndBallLinf implementation")
print()

# Test parameters
r = 2.0
print(f"L-infinity ball radius r = {r}")
print(f"Ball: {{x : ||x||_inf <= {r}}}")
print()

# Create indicator function
indicator = IndBallLinf(r=r)

# Test 1: Point inside ball
print("Test 1: Point inside ball")
x_inside = np.array([0.5, -1.0, 1.5, -0.3])
inf_norm = np.max(np.abs(x_inside))
print(f"x = {x_inside}")
print(f"||x||_inf = {inf_norm} (should be <= {r})")
print(f"eval(x) = {indicator.eval(x_inside)}")
prox_value = indicator.prox(x_inside.copy(), gamma=1.0)
y_prox = np.clip(x_inside, -r, r)
print(f"prox() returned (indicator value): {prox_value}")
print(f"Projected point should be x: {np.allclose(x_inside, y_prox)}")
print(f"eval(projected point) = {indicator.eval(y_prox)}")
print()

# Test 2: Point outside ball (all components > r)
print("Test 2: Point outside ball (all components > r)")
x_outside = np.array([3.0, 4.0, 2.5, 3.5])
inf_norm_outside = np.max(np.abs(x_outside))
print(f"x = {x_outside}")
print(f"||x||_inf = {inf_norm_outside} (should be > {r})")
print(f"eval(x) = {indicator.eval(x_outside)}")
prox_value_outside = indicator.prox(x_outside.copy(), gamma=1.0)
y_prox_outside = np.clip(x_outside, -r, r)
print(f"prox() returned (indicator value): {prox_value_outside}")
print(f"Projected point = {y_prox_outside}")
print(f"||prox(x)||_inf = {np.max(np.abs(y_prox_outside))} (should be <= {r})")
print(f"Projection formula correct: {np.allclose(y_prox_outside, np.clip(x_outside, -r, r))}")
print(f"eval(projected point) = {indicator.eval(y_prox_outside)}")
print()

# Test 3: Point on boundary
print("Test 3: Point on boundary")
x_boundary = np.array([r, -r, 0.0, r])
inf_norm_boundary = np.max(np.abs(x_boundary))
print(f"x = {x_boundary}")
print(f"||x||_inf = {inf_norm_boundary} (should be = {r})")
print(f"eval(x) = {indicator.eval(x_boundary)}")
prox_value_boundary = indicator.prox(x_boundary.copy(), gamma=1.0)
y_prox_boundary = np.clip(x_boundary, -r, r)
print(f"prox() returned (indicator value): {prox_value_boundary}")
print(f"Projected point should be x: {np.allclose(x_boundary, y_prox_boundary)}")
print(f"eval(projected point) = {indicator.eval(y_prox_boundary)}")
print()

# Test 4: Point with mixed signs outside ball
print("Test 4: Point with mixed signs outside ball")
x_mixed = np.array([-3.0, 1.0, 2.5, -4.0])
inf_norm_mixed = np.max(np.abs(x_mixed))
print(f"x = {x_mixed}")
print(f"||x||_inf = {inf_norm_mixed} (should be > {r})")
print(f"eval(x) = {indicator.eval(x_mixed)}")
prox_value_mixed = indicator.prox(x_mixed.copy(), gamma=1.0)
y_prox_mixed = np.clip(x_mixed, -r, r)
print(f"prox() returned (indicator value): {prox_value_mixed}")
print(f"Projected point = {y_prox_mixed}")
print(f"||prox(x)||_inf = {np.max(np.abs(y_prox_mixed))} (should be <= {r})")
print(f"eval(projected point) = {indicator.eval(y_prox_mixed)}")
print()

# Test 5: Validation - negative radius
print("Test 5: Validation - negative radius")
try:
    bad_indicator = IndBallLinf(r=-1.0)
    print("ERROR: Should have raised ValueError for negative radius")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test 6: Validation - zero radius
print("Test 6: Validation - zero radius")
try:
    bad_indicator = IndBallLinf(r=0.0)
    print("ERROR: Should have raised ValueError for zero radius")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

# Test 7: Multiple random points
print("Test 7: Testing multiple random points")
np.random.seed(42)
for i in range(5):
    x_test = np.random.normal(size=4) * 3  # Scale to get some outside ball
    prox_val = indicator.prox(x_test.copy(), gamma=1.0)
    y_prox_test = np.clip(x_test, -r, r)
    inf_norm_test = np.max(np.abs(y_prox_test))
    print(f"  Point {i+1}: ||x||_inf = {np.max(np.abs(x_test)):.4f}, "
          f"||prox(x)||_inf = {inf_norm_test:.4f}, "
          f"in ball: {inf_norm_test <= r + 1e-10}, prox() value: {prox_val}")

print("\nAll tests completed!")

