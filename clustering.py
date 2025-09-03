from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_kmeans_clustering(tfidf_matrix, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(tfidf_matrix)
    print("KMeans clustering done")
    return kmeans.labels_

def print_cluster_samples(df, cluster_labels, n_clusters=10, n_samples=5):
    df = df.copy()
    df['cluster'] = cluster_labels
    for i in range(n_clusters):
        print(f"\nCluster {i}:")
        try:
            samples = df[df['cluster'] == i]['title'].sample(n_samples, random_state=42).to_list()
        except ValueError:
            samples = df[df['cluster'] == i]['title'].to_list()  # fallback if fewer than n_samples
        print(samples)

def plot_cluster_distribution(df_filtered):
    sns.set(style="whitegrid")
    cluster_counts = df_filtered['cluster'].value_counts().sort_index()

    # Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, color='steelblue')
    plt.title("Number of Shows in Each KMeans Cluster", fontsize=14)
    plt.xlabel("Cluster Number")
    plt.ylabel("Number of Shows/Movies")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(cluster_counts.values, labels=cluster_counts.index,
            autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("viridis", n_colors=len(cluster_counts)))
    plt.title("Distribution of Shows by Cluster", fontsize=14)
    plt.axis('equal')
    plt.show()