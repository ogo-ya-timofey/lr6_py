import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D


# Генерация базовых данных (2 признака через make_blobs)
X, y = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=0.7,
    random_state=0
)

# Масштабирование признаков под заданные диапазоны
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 1-й и 2-й параметры в нужных диапазонах
param1 = X_scaled[:, 0] * (5 - 0.1) + 0.1   # [0.1; 5]
param2 = X_scaled[:, 1] * (3 - 0.1) + 0.1   # [0.1; 3]

# 3-й категориальный параметр
param3 = np.random.choice([10, 20, 80, 90], size=200)

# Формирование DataFrame
df = pd.DataFrame({
    "param1": param1,
    "param2": param2,
    "param3": param3
})

print(df.head())


# Кластеризация K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
kmeans.fit(df)

print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
print(Counter(kmeans.labels_))


# -------------------- 3D ВИЗУАЛИЗАЦИЯ --------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# точки данных
ax.scatter(
    df["param1"],
    df["param2"],
    df["param3"],
    c=kmeans.labels_,
    cmap='viridis',
    s=40
)

# центроиды кластеров
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 2],
    marker='X',
    c='red',
    s=120,
    label='centroids'
)

ax.set_xlabel("Param 1")
ax.set_ylabel("Param 2")
ax.set_zlabel("Param 3 (%)")

plt.legend()
plt.show()