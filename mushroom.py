import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#Read mushrooms data from repo
df=fetch_ucirepo(id=73)
print(df)

#Get features
X=df.data.features
#print(pd.isna(X).sum())
#Features contain 2480 of NaNs in stalk-root. Remove bad column:
X=X.drop(labels="stalk-root",axis=1)
#print(pd.isna(X).sum())
#print(X)

#Get targets
y=df.data.targets
#print(y)

#Get metadata
meta=df.metadata
v_info=df.variables

#Transform from categorical to numerical
LE=LabelEncoder()
y=LE.fit_transform(y)
#print(y)
X=pd.DataFrame({col: LE.fit_transform(X[col]) for col in X}, index=X.index)
#print(X)

#Scale all data with MinMax to 0-1
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
print(X)
