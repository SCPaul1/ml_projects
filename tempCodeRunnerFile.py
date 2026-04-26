import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# DATA  (same example from the lesson)
# ─────────────────────────────────────────────
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 5, 4])

n = len(x)

# ─────────────────────────────────────────────
# METHOD 1: FROM SCRATCH  (Normal Equations)
#   β̂ = (XᵀX)⁻¹ Xᵀy
# ─────────────────────────────────────────────

# Build design matrix X  (column of 1s  +  x column)
X = np.column_stack([np.ones(n), x])   # shape (4, 2)
print("Design matrix X:")
print(X)

# Compute XᵀX and Xᵀy
XtX = X.T @ X          # (2,2)
Xty = X.T @ y          # (2,)

print("\nXᵀX =\n", XtX)
print("Xᵀy =", Xty)

# Invert XᵀX  and solve for β̂
beta_hat = np.linalg.inv(XtX) @ Xty

beta0, beta1 = beta_hat
print(f"\n── From Scratch ──")
print(f"β̂₀ (intercept) = {beta0:.4f}")
print(f"β̂₁ (slope)     = {beta1:.4f}")
print(f"Fitted line:  ŷ = {beta0:.2f} + {beta1:.2f}x")

# Fitted values and residuals
y_hat = X @ beta_hat
residuals = y - y_hat
sse = np.sum(residuals ** 2)   # Sum of Squared Errors

print(f"\nFitted values ŷ : {y_hat}")
print(f"Residuals  e=y-ŷ: {residuals.round(4)}")
print(f"SSE (minimized) : {sse:.4f}")

# ─────────────────────────────────────────────
# METHOD 2: NumPy built-in  (lstsq solver)
#   More numerically stable than inverting XᵀX
# ─────────────────────────────────────────────
beta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
print(f"\n── NumPy lstsq ──")
print(f"β̂₀ = {beta_lstsq[0]:.4f},  β̂₁ = {beta_lstsq[1]:.4f}")

# ─────────────────────────────────────────────
# METHOD 3: scikit-learn  LinearRegression
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)   # sklearn expects 2D X
print(f"\n── scikit-learn ──")
print(f"β̂₀ = {model.intercept_:.4f},  β̂₁ = {model.coef_[0]:.4f}")
print(f"R²  = {model.score(x.reshape(-1,1), y):.4f}")

# ─────────────────────────────────────────────
# PREDICT for a new value
# ─────────────────────────────────────────────
x_new = 5
y_pred = beta0 + beta1 * x_new
print(f"\nPrediction at x={x_new}: ŷ = {y_pred:.2f}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
x_line = np.linspace(0.5, 5, 100)
y_line = beta0 + beta1 * x_line

plt.figure(figsize=(7, 5))
plt.scatter(x, y, color='coral', s=80, zorder=5, label='Observed data')
plt.scatter(x, y_hat, color='steelblue', s=50, zorder=4,
            marker='o', facecolors='none', linewidths=1.5, label='Fitted values')
plt.plot(x_line, y_line, color='steelblue', linewidth=2,
         label=f'ŷ = {beta0:.2f} + {beta1:.2f}x')

# Draw residuals
for xi, yi, yhi in zip(x, y, y_hat):
    plt.plot([xi, xi], [yi, yhi], color='coral',
             linestyle='--', linewidth=1.2, alpha=0.7)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('least_squares_plot.png', dpi=150)
plt.show()
print("\nPlot saved to least_squares_plot.png")