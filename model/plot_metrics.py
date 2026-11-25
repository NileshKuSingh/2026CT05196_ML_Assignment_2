from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).resolve().parent
csv_file = here / "metrics_summary.csv"
out_img = here / "metrics_comparison.png"

# Reads the saved evaluation table from Step 3
scores_df = pd.read_csv(csv_file)

metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
plot_df = scores_df.set_index("ML Model Name")[metric_cols]

# Grouped bar chart makes comparing every metric per model easy at a glance
ax = plot_df.plot(kind="bar", figsize=(11, 6), width=0.78)
ax.set_title("Classification Model Metrics - Student Dropout Dataset (UCI 697)")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(out_img, dpi=200)
print("Saved chart to:", out_img)