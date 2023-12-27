import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

#Read mushrooms data from the repo
#https://archive.ics.uci.edu/dataset/73/mushroom
df=fetch_ucirepo(id=73)
#print(df)

#Get features
X=df.data.features
LE=LabelEncoder()
X=pd.DataFrame({col: LE.fit_transform(X[col]) for col in X}, index=X.index)

inertias = []

for i in range(1,31):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,31), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()