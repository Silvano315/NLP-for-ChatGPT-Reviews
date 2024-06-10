import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plots a bar chart for a specified numerical feature 
def plot_bar(df, feature, num_bars_max=None):

    value_counts = df[feature].value_counts()
    
    if num_bars_max is not None:
        value_counts = value_counts.head(num_bars_max)
    
    total = value_counts.sum()
    percentages = (value_counts / total) * 100

    cmap = plt.get_cmap('viridis', len(value_counts))
    colors = cmap(np.arange(len(value_counts)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index, value_counts, color=colors)

    legend_labels = [f'{index}: {count} ({percentage:.2f}%)' for index, count, percentage in zip(value_counts.index, value_counts, percentages)]
    plt.legend(bars, legend_labels, title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
