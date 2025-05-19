import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

def cal_near(df_new):
    neighbours = NearestNeighbors(n_neighbors=25)
    nbrs = neighbours.fit(df_new)
    dis, ind = nbrs.kneighbors(df_new)
    dis = np.sort(dis, axis=0)
    dis = dis[:, -1]

    plt.figure(figsize=(8, 8))
    plt.plot(dis)
    plt.show()

def get_clusters(data):
    # cal_near(data)
    cluster = DBSCAN(eps=0.03, min_samples=25)
    data['Label'] = cluster.fit_predict(data)
    plt.figure(figsize=(15, 12))
    # sns.scatterplot(x='X', y='Y', data=data, hue='Label', palette='tab10', s=200)
    # sns.lineplot(x='X',
    #              y='Y',
    #              data=data
    #              )
    plt.plot(data['X'], data['Y'],)
    plt.scatter(data['X'], data['Y'], c=data['Label'], cmap='tab10')
    plt.legend()


data_source = pd.read_csv("data1.csv")
get_clusters(data_source)
# plt.figure(figsize=(15, 12))
data_source = pd.read_csv("data.csv")
get_clusters(data_source)
plt.show()