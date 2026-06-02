# Импорт библиотек для генерации данных, кластеризации, анализа и визуализации.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sb

# Пример построения простого линейного графика средствами Matplotlib.
# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [1, 4, 2, 5])
# plt.ylabel('some numbers')
# plt.show()

# Генерация набора данных из 4 кластеров и сохранение его в DataFrame.
dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
print(df.head(2))

# Определение оптимального количества кластеров методом локтя.
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1, 12), force_model=True).fit(df)
# visualizer.show()

# Обучение модели K-Means и вывод её основных параметров.
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)

# Подсчёт количества объектов в каждом кластере.
print(Counter(kmeans.labels_))

# Визуализация кластеров и отображение центров кластеризации.
sb.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', c='r', s=80,label='centroids')
plt.legend()
plt.show()
