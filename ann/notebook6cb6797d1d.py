# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import pandas as pd
import numpy as np
data=pd.read_csv('./fetal_health.csv')
data.head()

# %%
data=data.drop_duplicates()
data.info()

# %%
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data['fetal_health']=enc.fit_transform(data['fetal_health'])

# %%
data['fetal_health'].value_counts()
y=data['fetal_health']
x=data.drop(['fetal_health'],axis=1)

# %%
for column in x.columns:
    x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min()) 
x.head()

# %%
from keras.utils.np_utils import to_categorical
y_cat=to_categorical(y,num_classes=3)

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x.values,y_cat,test_size=0.1,stratify=y)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# %%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
dl_model = Sequential() 

dl_model.add(Dense(256,  activation = 'relu' ,input_shape=([21]))) #input layer
dl_model.add(Dense(512,  activation = 'relu'))
dl_model.add(Dense(3,activation = 'softmax'))
dl_model.summary()
dl_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' ,metrics = ['accuracy','Precision','Recall','AUC'])

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
filepath='./kaggle_model.h5'
try:
    dl_model = load_model(filepath)
except:
    pass
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
num_epochs = 2000
history = dl_model.fit(xtrain ,
                    ytrain ,
                    batch_size=32,
                    epochs= num_epochs ,
                    steps_per_epoch=50,
                    validation_data=(xtest ,ytest),callbacks=checkpoint)

# %%
dl_model.evaluate(xtest ,ytest)

# %%
dl_model.evaluate(xtrain,ytrain)

# %%
dl_model.predict([[134.0,0.001,0.0,0.013,0.008,0.0,0.003,29.0,6.3,0.0,0.0,150.0,50.0,200.0,6.0,3.0,71.0,107.0,106.0,215.0,0.0]])

# %%
dl_model.save("./kaggle_model.h5")


