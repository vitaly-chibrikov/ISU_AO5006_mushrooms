from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

#Read mushrooms data from the repo
#https://archive.ics.uci.edu/dataset/73/mushroom
df=fetch_ucirepo(id=73)
#print(df)

#Get features
X=df.data.features
#Fix NaNs
#Features contain 2480 of NaNs in stalk-root. Remove bad column:
X=X.drop(labels="stalk-root",axis=1)
print("X shape after bad column removal: ", X.shape)
print(X)

inertias = []
for col in X:
    print(col, ": ", pd.Series(list(X[col])).unique())
    print(pd.Series(list(X[col])).value_counts())
    inertias.append(1)
    
columns_number=X.shape[1]
plt.plot(range(0,columns_number), inertias, marker='o')
plt.title('KMeans inertias')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()   