import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_and_plot(log_file_path):
    pattern = r"\[Seed (\d+)\] Test (All|Head|Tail) F1 Score: ([\d.]+)"
    
    data = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                seed = int(match.group(1))
                category = match.group(2)
                F1 = float(match.group(3))
                data.append({"Seed": seed, "Category": category, "F1": F1})
    
    df = pd.DataFrame(data)
    if df.empty:
        print("No matching data found in the log. Please check the log format or path.")
        return

    # Pivot table: rows are Seed, columns are Category
    pivot_df = df.pivot(index='Seed', columns='Category', values='F1')
    # Ensure column order
    pivot_df = pivot_df[['All', 'Head', 'Tail']]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot_df.plot(kind='line', marker='o', ax=ax, linewidth=2, markersize=8)

    plt.title('Model Sensitivity to Random Seeds (F1)', fontsize=14, fontweight='bold')
    plt.xlabel('Random Seed', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(pivot_df.index) # Ensure X-axis only shows existing seeds
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='User Group', frameon=True)

    for col in pivot_df.columns:
        for x, y in zip(pivot_df.index, pivot_df[col]):
            ax.annotate(f'{y:.3f}', xy=(x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9)

    y_min = df['F1'].min() * 0.98
    y_max = df['F1'].max() * 1.02
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig('seed_sensitivity.png', dpi=300)
    print("The chart has been saved as seed_sensitivity.png")

if __name__ == "__main__":
    parse_log_and_plot('runs/explore/smia_seed/train.log')