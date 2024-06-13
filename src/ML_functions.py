from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.under_sampling import ClusterCentroids, TomekLinks, RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score



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

# Function for Cluster Centroids Undersampling
def cluster_centroids_undersample(X, y):

    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_resample(X, y)

    return X_res, y_res

#Function for Tomek Links method
def tomek_links_undersample(X, y):

    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)

    return X_res, y_res


#Function for Random Undersampling
def random_undersample(X, y):
    
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    return X_res, y_res


#Function to perform feature reduction and visualize clusters
def visualize_clusters(X, y, method='PCA'):

    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
    
    X_reduced = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization using {method}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

#Function to calculate Silhouette Score
def calculate_silhouette_score(X, y):
    return silhouette_score(X, y)