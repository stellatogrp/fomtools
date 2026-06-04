"""
Plot Lasso L2O (learned stepsizes) worst-case bounds (MIP vs PEPit vs
simulation) from the CSV produced by comparison_lasso_l2o.py.

Usage:
    python plot_lasso_l2o_learned.py [results.csv]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else "lasso_l2o_learned_results.csv"
data = np.genfromtxt(csv_path, delimiter=",", names=True)

iters = data["k"]
mip_residuals = data["mip"]
pepit_residuals = data["pepit"]
sim_residuals = data["sim"]
R = float(data["R"][0])
num_star = int(data["num_star"][0])
K = int(iters[-1])

plt.figure(figsize=(7, 4))
plt.plot(iters, mip_residuals,       "o-",  label="MIP")
plt.plot(iters, R * pepit_residuals, "^:",  label="PEPit")
plt.plot(iters, sim_residuals,       "d-.", label=f"Simulation (max over N={num_star})")
plt.yscale("log")
plt.xlabel(r"Iteration $k$")
plt.ylabel(r"Worst-case $\|H(s^k - s^{k-1})\|_\infty$")
plt.title(f"Lasso L2O (learned stepsizes): MIP vs PEPit ({K} iterations)")
plt.legend()
plt.tight_layout()
plt.show()
