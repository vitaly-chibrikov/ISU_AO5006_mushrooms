import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

#from keras.utils import np_utils
from keras import utils as np_utils

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
#print(X)

#Get targets
y=df.data.targets
print("y shape: ", y.shape)
#print(y)

#Get metadata
meta=df.metadata
v_info=df.variables

################ Prepare the data ################

#Fix NaNs
print(pd.isna(X).sum())
#Features contain 2480 of NaNs in stalk-root. Remove bad column:
X=X.drop(labels="stalk-root",axis=1)
print("X shape after bad column removal: ", X.shape)

#Transform from categorical to numerical
LE=LabelEncoder()
y=LE.fit_transform(y.values.ravel())
y=np_utils.to_categorical(y)
print(y)
X=pd.DataFrame({col: LE.fit_transform(X[col]) for col in X}, index=X.index)

'''
#Fix unbalanced data
over_sampling=SMOTE()
X,y=over_sampling.fit_resample(X, y)
print("X shape after over sampling fix: ", X.shape)
print("y shape after over sampling fix: ", y.shape)
print(X)
print(y)
print("-----------------------------------")
'''

#Check for outliers
clf=LocalOutlierFactor()
outlier=clf.fit_predict(X)
print("Otliers found:", (outlier == -1).sum())
#Remove outliers
X=X[outlier == 1]
print("X shape after otliers removal: ", X.shape)
y=y[outlier == 1]
print("y shape otliers removal: ", y.shape)

'''
#Check variance and apply dimension reduction
pca=PCA()
pca.fit(X)
print("Explained variance: ", pca.explained_variance_ratio_.cumsum())
#It looks like we can lower the dimension to 12 without losing 2% of variance
pca=PCA(n_components=12)
X=pca.fit_transform(X)
print("X shape after dimension reduction: ", X.shape)
'''

#Scale all data with MinMax to 0-1
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
print("Prepared data:")
print(X)
print(y)

################ Split the data ################

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.33, random_state=123)

################ Prepare the model #########

model=Sequential()
model.add(Dense(50,input_dim=21, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))

model.summary()

################ Learn from the data ###########

tf.keras.utils.plot_model(model,"exp.png",
                          show_shapes=True,
                          show_layer_names=True)

model.compile(loss="categorical_crossentropy",
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