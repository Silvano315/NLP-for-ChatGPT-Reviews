from matplotlib import pyplot as plt
import numpy as np

# Function to visualize models metrics
def plot_model_metrics(model_metrics, models, metrics):

    final_metrics = {name: {metric: (np.mean(scores), np.std(scores)) for metric, scores in metrics_dict.items()} for name, metrics_dict in model_metrics.items()}

    colors = {
        'RandomForest': 'lightgreen',
        'XGBoost': 'lightblue',
        'LogisticRegression': 'salmon',
        'SVM' : 'lightyellow'
    }

    fig, axs = plt.subplots(2, 3, figsize=(20, 12)) 

    axs = axs.flatten()

    for i, metric in enumerate(metrics):

        if metric == 'confusion_matrix':
            continue

        means = [final_metrics[name][metric][0] for name in models]
        stds = [final_metrics[name][metric][1] for name in models]
        model_names = list(models.keys())
        colors_list = [colors[name] for name in model_names]
        
        axs[i].bar(model_names, means, yerr=stds, capsize=5, color=colors_list)
        axs[i].set_title(f'{metric.capitalize()} Comparison')
        axs[i].set_xlabel('Model')
        axs[i].set_ylabel(f'Mean {metric.capitalize()}')
        axs[i].set_ylim(0, 1) 
    
    for j in range(len(metrics), len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()