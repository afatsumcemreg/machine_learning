# Importing libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
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
df[0:5]

# kmeans modeling
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

# Getting cluster numbers
kmeans.n_clusters

# Getting cluster centers
kmeans.cluster_centers_

# Getting cluster labels
kmeans.labels_

# Calculation error value
kmeans.inertia_

# k_means_modeling and application
k_means_model = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    k_means_model = KMeans(n_clusters=k).fit(df)
    ssd.append(k_means_model.inertia_)

# Visualization of ssd variables
plt.plot(K, ssd, 'bx-')
plt.xlabel('SSD values for different k values')
plt.title('Elbow method for optimum cluster number')
plt.show(block=True)

# Determining optimum k value using Elbow method
k_means_model = KMeans()
elbow = KElbowVisualizer(k_means_model, k=(2, 20))
elbow.fit(df)
plt.show(block=True)
elbow.elbow_value_

# kmeans final model
# Creating final clusters
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kmeans.n_clusters
kmeans.labels_

# determining which observation unit belongs to which cluster
clusters = kmeans.labels_
df = pd.read_csv('01_miuul_machine_learning_summercamp/00_datasets/USArrests.csv', index_col=0)
df['Cluster'] = clusters
df.head()
df['Cluster'] = df['Cluster'] + 1
df.head()

# determine which state belongs to which cluster
df[df['Cluster'] == 1]

df.groupby('Cluster').agg(['count', 'mean', 'median'])
df.to_csv('clusters.csv')