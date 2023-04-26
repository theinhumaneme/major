# %%
import numpy as np
import pandas as pd

# %%
dataset = pd.read_csv("./fetal_health.csv")

# %%
dataset.info()

# %%
dataset.isnull().any()

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# %%
x = dataset.iloc[:,0:21].values
y = dataset.iloc[:,21:22].values
# y
# x.shape

# %%
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# %%
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.fit_transform(X_test)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# %%
model.add(Dense(units = 21, activation = "relu", kernel_initializer = "random_uniform"))

# %%
model.add(Dense(units = 21, activation = "relu", kernel_initializer = "random_uniform"))

# %%
model.add(Dense(units = 42, activation = "relu", kernel_initializer = "random_uniform"))

# %%
model.add(Dense(units = 42, activation = "relu", kernel_initializer = "random_uniform"))

# %%
model.add(Dense(units = 21, activation = "relu", kernel_initializer = "random_uniform"))

# %%
model.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "random_uniform"))

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
model.fit(X_train,Y_train, batch_size=32, epochs=2000)

# %%
model.save(filepath="./model.h5")


