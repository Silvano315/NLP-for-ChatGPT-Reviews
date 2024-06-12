from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


# Function to visualize models metrics
def plot_model_metrics(model_metrics, models, metrics):

    final_metrics = {name: {metric: (np.mean(scores), np.std(scores)) for metric, scores in metrics_dict.items()} for name, metrics_dict in model_metrics.items()}

    colors = {
        'RandomForest': 'lightgreen',
        'XGBoost': 'lightblue',
        'LogisticRegression': 'salmon'    }

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


# Plot for confusion matrix for each model
def plot_confusion_matrix(conf_matrix, model_name):

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')

    plt.xticks(np.arange(5) + 0.5, np.arange(1, 6))
    plt.yticks(np.arange(5) + 0.5, np.arange(1, 6), rotation = 0)

    plt.show()