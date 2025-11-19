import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load SVR tuning results
svr_results = pd.read_csv("svr_grid_results.csv")

# Convert negative MSE to positive MSE
svr_results["test_MSE"] = -svr_results["mean_test_score"]

# ==== Heatmap: C vs epsilon ====
pivot_svr = svr_results.pivot_table(
    values="test_MSE",
    index="param_svr__C",
    columns="param_svr__epsilon"
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_svr, annot=True, cmap="magma")
plt.title("SVR Test MSE Heatmap")
plt.xlabel("epsilon")
plt.ylabel("C")
plt.tight_layout()
plt.savefig("svr_heatmap.png", dpi=300)
plt.show()

# ==== Line Plot: Test MSE for different epsilon values ====
plt.figure(figsize=(8, 6))
for eps in svr_results["param_svr__epsilon"].unique():
    subset = svr_results[svr_results["param_svr__epsilon"] == eps]
    plt.plot(
        subset["param_svr__C"],
        subset["test_MSE"],
        marker='o',
        label=f"epsilon={eps}"
    )

plt.title("SVR Test MSE by Hyperparameters")
plt.xlabel("C")
plt.ylabel("Test MSE")
plt.legend()
plt.tight_layout()
plt.savefig("svr_lineplot.png", dpi=300)
plt.show()
