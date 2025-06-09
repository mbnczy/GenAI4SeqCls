import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def density_plot_with_descstat(dataset, save_as, inst_col = "input_ids", label_col = "label"):
    input_lengths = [len(input_ids) for input_ids in dataset['input_ids']]
    lengths = np.array(input_lengths)
    
    desc_stats = {
        "Count": len(lengths),
        "Mean": np.mean(lengths),
        "Std Dev": np.std(lengths),
        "Min": np.min(lengths),
        "25%": np.percentile(lengths, 25),
        "Median": np.median(lengths),
        "75%": np.percentile(lengths, 75),
        "Max": np.max(lengths),
    }
    
    plt.figure(figsize=(12, 7))
    sns.kdeplot(lengths, fill=True, color="skyblue", linewidth=2)
    plt.title("Density Plot of Instruction Lengths")
    plt.xlabel("Instruction Length")
    plt.ylabel("Density")
    plt.grid(True)
    
    plt.axvline(desc_stats["Mean"], color='red', linestyle='--', label=f"Mean: {desc_stats['Mean']:.2f}")
    plt.axvline(desc_stats["Median"], color='green', linestyle='--', label=f"Median: {desc_stats['Median']:.2f}")
    plt.axvline(desc_stats["25%"], color='orange', linestyle=':', label=f"25%: {desc_stats['25%']:.2f}")
    plt.axvline(desc_stats["75%"], color='purple', linestyle=':', label=f"75%: {desc_stats['75%']:.2f}")
    
    stat_text = "\n".join([
        f"Count: {desc_stats['Count']}",
        f"Std Dev: {desc_stats['Std Dev']:.2f}",
        f"Min: {desc_stats['Min']:.2f}",
        f"Max: {desc_stats['Max']:.2f}"
    ])
    
    plt.gca().text(0.98, 0.95, stat_text,
                   horizontalalignment='right',
                   verticalalignment='top',
                   transform=plt.gca().transAxes,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as, dpi=600)


def density_plot_per_label(dataset, save_as, inst_col = "input_ids", label_col = "label"):
    lengths_by_label = [
        {"label": example[label_col], "input_length": len(example[inst_col])}
        for example in dataset
    ]
    df = pd.DataFrame(lengths_by_label)
    
    unique_labels = sorted(df['label'].unique())
    palette = sns.color_palette("husl", len(unique_labels))
    label_color_map = dict(zip(unique_labels, palette))
    
    plt.figure(figsize=(12, 7))
    
    sns.kdeplot(
        data=df, 
        x="input_length", 
        hue="label", 
        fill=True, 
        alpha=0.1, 
        linewidth=2,
        palette=label_color_map
        
    )
    
    plt.title("Density of Instruction Lengths by Label")
    plt.xlabel("Intruction Length")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as, dpi=600)

