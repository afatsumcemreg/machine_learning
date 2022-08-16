# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


## Importing dataset
df = pd.read_csv("01_miuul_machine_learning_summercamp/00_datasets/USArrests.csv", index_col=0)
df.head()

# Checking the dataset
df.isnull().sum()
df.info()
df.describe().T

# Standardization
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# Linakge method via euclidean distance
hc_average = linkage(df, 'average')

# Dendogram figure
plt.figure(figsize=(10, 5))
plt.title('Hierarcial Cluster Method')
plt.xlabel('Observation unit')
plt.ylabel('Distances')
dendrogram(hc_average, leaf_font_size=10)
plt.show(block=True)

# Dendogram figure to obtain lesser observation
plt.figure(figsize=(10, 5))
plt.title('Hierarcial Cluster Method')
plt.xlabel('Observation Unit')
plt.ylabel('Distances')
dendrogram(hc_average, truncate_mode='lastp', p=10, show_contracted=True, leaf_font_size=10)
plt.show(block=True)

# Determining the cluster number (4)
plt.figure(figsize=(8, 5))
plt.title('Dendograms')
dend = dendrogram(hc_average, truncate_mode='lastp', p=10, show_contracted=True, leaf_font_size=10)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.show(block=True)

# Final HCA model
cluster = AgglomerativeClustering(n_clusters=4, linkage='average')
clusters = cluster.fit_predict(df)

# Re-reed the data set
df = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/USArrests.csv', index_col=0)
df['hc_cluster_no'] = clusters
df['hc_cluster_no'] = df['hc_cluster_no'] + 1
df.head()

# States being hc_cluster_no is between 1 and 7
len(df[df['hc_cluster_no'] == 1])
len(df[df['hc_cluster_no'] == 2])
len(df[df['hc_cluster_no'] == 3])
len(df[df['hc_cluster_no'] == 4])
len(df[df['hc_cluster_no'] == 5])
len(df[df['hc_cluster_no'] == 6])
len(df[df['hc_cluster_no'] == 7])