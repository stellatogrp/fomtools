# Code Generation Speedup Results

Results from the Marimo code generation speedup examples. Each timing is for one complete solve with 200 iterations.

| Example | Direct Python | Generated C after code generation | Generated C after C optimization | Speedup from code generation | Speedup from C optimization | Total speedup |
|---|---:|---:|---:|---:|---:|---:|
| Lasso | `1053.063 us` | `7.335 us` | `1.546 us` | `143.6x` | `4.75x` | `681.3x` |
| Optimal control | `2817.667 us` | `153.596 us` | `29.704 us` | `18.3x` | `5.17x` | `94.9x` |
| Network flow | `2358.812 us` | `10.613 us` | `2.593 us` | `222.3x` | `4.09x` | `909.8x` |
| Nonnegative least squares | `690.375 us` | `9.352 us` | `1.459 us` | `73.8x` | `6.41x` | `473.3x` |
| Portfolio | `2017.354 us` | `14.831 us` | `6.351 us` | `136.0x` | `2.34x` | `317.6x` |

## Verification

All optimized C solvers were kept after verification and timing.

| Example | Final verification result | Worst max-absolute error |
|---|---|---:|
| Lasso | `VERIFY OK` | `0.000e+00` |
| Optimal control | `VERIFY OK` | `5.031e-16` |
| Network flow | `VERIFY OK` | `1.443e-14` |
| Nonnegative least squares | `VERIFY OK` | `1.388e-17` |
| Portfolio | `VERIFY OK` | `2.220e-16` |
