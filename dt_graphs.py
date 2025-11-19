import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Decision Tree tuning results
dt_results = pd.read_csv("dt_grid_results.csv")

# Convert negative MSE to positive MSE
dt_results["test_MSE"] = -dt_results["mean_test_score"]

# ==== Heatmap: max_depth vs min_samples_split ====
pivot_dt = dt_results.pivot_table(
    values="test_MSE",
    index="param_max_depth",
    columns="param_min_samples_split"
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_dt, annot=True, cmap="viridis")
plt.title("Decision Tree Test MSE Heatmap")
plt.xlabel("min_samples_split")
plt.ylabel("max_depth")
plt.tight_layout()
plt.savefig("dt_heatmap.png", dpi=300)
plt.show()

# ==== Line Plot showing test MSE for each max_depth ====
plt.figure(figsize=(8, 6))
for depth in dt_results["param_max_depth"].unique():
    subset = dt_results[dt_results["param_max_depth"] == depth]
    plt.plot(
        subset["param_min_samples_split"],
        subset["test_MSE"],
        marker='o',
        label=f"max_depth={depth}"
    )

plt.title("Decision Tree Test MSE by Hyperparameters")
plt.xlabel("min_samples_split")
plt.ylabel("Test MSE")
plt.legend()
plt.tight_layout()
plt.savefig("dt_lineplot.png", dpi=300)
plt.show()
