import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# To fix errors of versions compatibility of SMOTE
# use "pip install scikit-learn==1.2.2"
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

import matplotlib.pyplot as plt

################ Load the data ###################

#Read mushrooms data from the repo
#https://archive.ics.uci.edu/dataset/73/mushroom
df=fetch_ucirepo(id=73)
#print(df)

#Get features
X=df.data.features
print("Original X shape: ", X.shape)
print(X)

#Get targets
y=df.data.targets
print("y shape: ", y.shape)
print(y)

#Get metadata
meta=df.metadata
v_info=df.variables

################ Prepare the data ################

#Fix NaNs
print(pd.isna(X).sum())
#Features contain 2480 of NaNs in stalk-root. Remove bad column:
X=X.drop(labels="stalk-root",axis=1)
print("X shape after NaN column removal: ", X.shape)
for col in X:
    print(col, ": ", pd.Series(list(X[col])).unique())
#Column "veil-type" contains only one value ['p'] (partial=p,universal=u)
X=X.drop(labels="veil-type",axis=1)
print("X shape after useless column removal: ", X.shape)

#Transform from categorical to numerical
LE=LabelEncoder()
y=LE.fit_transform(y.values.ravel())
X=pd.DataFrame({col: LE.fit_transform(X[col]) for col in X}, index=X.index)

#Check for outliers
clf=LocalOutlierFactor()
outlier=clf.fit_predict(X)
print("Otliers found:", (outlier == -1).sum())
#Remove outliers
X=X[outlier == 1]
print("X shape after otliers removal: ", X.shape)
y=y[outlier == 1]
print("y shape after otliers removal: ", y.shape)

#Fix unbalanced data
#Synthetic Minority Over-sampling Technique 
print("y balance before SMOTE: ")
print(pd.Series(list(y)).value_counts())
over_sampling=SMOTE()
X,y=over_sampling.fit_resample(X, y)
print("X shape after balancing: ", X.shape)
print("y shape after balancing: ", y.shape)
print("y balance after SMOTE: ")
print(pd.Series(list(y)).value_counts())

#Check variance and apply dimension reduction
#Principal Component Analysis
pca=PCA()
pca.fit(X)
print("Explained variance: ", pca.explained_variance_ratio_.cumsum())
#It looks like we can lower the dimension to 19 without losing 0.0001% of variance
pca=PCA(n_components=19)
X=pca.fit_transform(X)
print("X shape after dimension reduction: ", X.shape)

#Scale all data with MinMax to 0-1
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
print("Prepared data:")
print(X)
print(y)

################ Split the data ################

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.33, random_state=0)

################ Prepare the model #########

model=Sequential()
model.add(Dense(50,input_dim=19, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))

model.summary()

################ Learn from the data ###########

tf.keras.utils.plot_model(model,"exp.png",
                          show_shapes=True,
                          show_layer_names=True)

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history= model.fit(
    X_train,
    Y_train,
    epochs=20,
    verbose=2,
    batch_size=16,
    validation_split=0.2
    )

################ Plot ################

plt.plot(history.history["accuracy"],"--")
plt.plot(history.history["val_accuracy"])
plt.title("Training performance")
plt.ylabel("Accuracy")
plt.xlabel("Cycle(epoch)")
plt.legend(["training","verification"],loc="lower right")
plt.show()

plt.plot(history.history["loss"],"--")
plt.plot(history.history["val_loss"])
plt.title("Model errors")
plt.ylabel("Errors")
plt.xlabel("Cycle(epoch)")
plt.legend(["training","verification"],loc="upper right")
plt.show()

scores=model.evaluate(X_test,Y_test)

print("Scores: ", scores)
#'''