import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# To fix errors of versions compatibility of SMOTE
# use "pip install scikit-learn==1.2.2"
from imblearn.over_sampling import SMOTE

################ Load the data ###################

#Read mushrooms data from repo
df=fetch_ucirepo(id=73)
#print(df)

#Get features
X=df.data.features
print("Original X shape: ", X.shape)
#print(pd.isna(X).sum())
#Features contain 2480 of NaNs in stalk-root. Remove bad column:
X=X.drop(labels="stalk-root",axis=1)
#print(pd.isna(X).sum())
print("X shape after bad column removal: ", X.shape)

#Get targets
y=df.data.targets
print("y shape: ", y.shape)

#Get metadata
meta=df.metadata
v_info=df.variables

################ Prepare the data ################

#Transform from categorical to numerical
LE=LabelEncoder()
y=LE.fit_transform(y.values.ravel())
#print(y)
X=pd.DataFrame({col: LE.fit_transform(X[col]) for col in X}, index=X.index)
#print(X)

#Fix unballanced data
over_sampling=SMOTE()
X,y=over_sampling.fit_resample(X, y)
print("X shape after over sampling: ", X.shape)
print("y shape after over sampling: ", y.shape)

#Check for outliers
clf=LocalOutlierFactor()
outlier=clf.fit_predict(X)
print("Otliers found:", (outlier == -1).sum())
#Remove outliers
X=X[outlier == 1]
print("X shape after otliers removal: ", X.shape)
y=y[outlier == 1]
print("y shape otliers removal: ", y.shape)

#Check variance and apply dimension reduction
pca = PCA()
pca.fit(X)
print("explained variance: ", pca.explained_variance_ratio_.cumsum())
#It looks like we can lower the dimension to 12 without losing 2% of variance
pca = PCA(n_components=12)
X=pca.fit_transform(X)
print("X shape after dimension reduction: ", X.shape)

#Scale all data with MinMax to 0-1
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
print(X)

################ Split the data ################

# Todo

################ Prepare the model #########

# Todo

################ Learn from data ###########

# Todo
