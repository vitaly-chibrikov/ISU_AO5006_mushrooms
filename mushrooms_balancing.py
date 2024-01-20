from ucimlrepo import fetch_ucirepo
import pandas as pd

#Read mushrooms data from the repo
#https://archive.ics.uci.edu/dataset/73/mushroom
df=fetch_ucirepo(id=73)
#print(df)

#Get features
X=df.data.features
y=df.data.targets

for col in X:
    print(col, ": ", pd.Series(list(X[col])).unique())

print(y)
print((y == 'p').sum())
print('not ',(y == 'e').sum())

indexes_list=list()
for col in X:
    indexes_list.extend(pd.Series(list(X[col])).unique())

print(pd.Series(indexes_list).unique())
balance = pd.DataFrame(index=pd.Series(indexes_list).unique())

for col in X:
    print(col, ": ", pd.Series(list(X[col])).unique())
    print(pd.Series(list(X[col])).value_counts())
    balance[col]=pd.Series(list(X[col])).value_counts()

print("X shape: ", X.shape)

print("y balance: ")
print(pd.Series(list(y)).value_counts())


 