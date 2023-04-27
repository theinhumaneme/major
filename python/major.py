# %% [markdown]
# # Fetal Health Classification using AI/ML E-13

# %% [markdown]
# ### Importing inital libraries

# %%
import warnings
warnings.filterwarnings('ignore')
import joblib
import seaborn as sns

#library for data preprocessing & to read CSV files
import pandas as pd

# used for linear algebra and multi-dimensional matrix manipulation
import numpy as np 

import os

#used for data visualisation
import matplotlib.pyplot as plt

#matplotlib is used internally, easier to use
import seaborn as sb

# %% [markdown]
# ### Loading and viewing the dataset

# %%
data_frame = pd.read_csv("./fetal_health.csv")
data_frame.head()

# %% [markdown]
# # About the data
# 
# This dataset records of features extracted from Cardiotocogram (**CTG**) exams, which were then classified by expert *obstetrician* into 3 classes: 
# - **Normal**
# - **Suspect**
# - **Pathological**
# ---
# The Dataset has the following features/attributes:
# 
# - **baseline value**: Baseline Fetal Heart Rate (FHR) (beats per minute)
# - **accelerations**: Number of accelerations per second
# - **fetal_movement**: Number of fetal movements per second
# - **uterine_contractions**: Number of uterine contractions per second
# - **light_decelerations**: Number of light decelerations (LDs) per second
# - **severe_decelerations**: Number of severe decelerations (SDs) per second
# - **prolongued_decelerations**: Number of prolonged decelerations (PDs) per second
# - **abnormal_short_term_variability**: Percentage of time with abnormal short term variability
# - **mean_value_of_short_term_variability**: Mean value of short term variability
# - **percentage_of_time_with_abnormal_long_term_variability**: Percentage of time with abnormal long term variability
# - **mean_value_of_long_term_variability**: Mean value of long term variability
# - **histogram_width**: Width of histogram made using all values from a record
# - **histogram_min**: Histogram minimum value
# - **histogram_max**: Histogram maximum value
# - **histogram_number_of_peaks**: Number of peaks in the exam histogram
# - **histogram_number_of_zeroes**: Number of zeros in the exam histogram
# - **histogram_mode**: Histogram mode
# - **histogram_mean**: Histogram mean
# - **histogram_median**: Histogram median
# - **histogram_variance**: Histogram variance
# - **histogram_tendency**: Histogram tendency
# - **fetal_health**: Encoded as 1-Normal; 2-Suspect; 3-Pathological.

# %% [markdown]
# ### Retreiving information of the dataset 

# %%
data_frame.info()

# %% [markdown]
# ## DATA ANALYISIS & PRE-PROCESSING 

# %% [markdown]
# ### Statistical trends in the data

# %%
data_frame.describe().T

# %% [markdown]
# ### Evaluating if our data is imbalanaced or not by analysing the target output.

# %%
colours=["green","gold", "red"]
sb.countplot(data=data_frame, x="fetal_health",palette=colours)

# %% [markdown]
# #### We can see there is an imbalance in the data.

# %% [markdown]
# ### A Heatmap to analyze how related any two datapoints are (Co-relation Matrix)

# %%
correlation_matrix = data_frame.corr()
plt.figure(figsize=(15,15))
sb.heatmap(correlation_matrix, annot=True,center=0)
plt.savefig("co-relation.svg")

# %% [markdown]
# Based on the co-relation matrix we can see that
# 1. prolonged_decelerations
# 2. percentage of time with abnormal long term variabilty
# 3. mean value of long term variability
# 
# have higher co-relation with fetal health and hence these are the most important features that wil be essential in the prediction of the state of the fetus

# %%
cols=['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability']
for i in cols:
#     sb.swarmplot(x=data_frame["fetal_health"], y=data_frame[i], color="black", alpha=0.5,size=1.5)
#     sb.boxenplot(x=data_frame["fetal_health"], y=data_frame[i], palette=colours)
    plt.show()

# %%
# remove the target data from the dataset to visualise the the distribution of the attributtes
XData =data_frame.drop(["fetal_health"], axis=1)
# store the target data for future reference
YData = data_frame["fetal_health"]

# %% [markdown]
# ### Visualise the distribution of data BEFORE feature scaling

# %%
colors=["#483D8B","#4682B4", "#87CEFA"]
features=['baseline value', 'accelerations', 'fetal_movement','uterine_contractions', 'light_decelerations', 'severe_decelerations',
           'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
             'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']
plt.figure(figsize=(20,10))
sb.boxenplot(data = XData,palette = colors)
plt.xticks(rotation=60)
plt.show()
plt.savefig("before-feature-scaling.svg")

# %% [markdown]
# ### Visualise the distribution of data AFTER feature scaling

# %% [markdown]
# Import the standard scaler from the scikit-learn library to scale the data/features to make the data suitable for model training

# %%
# Performing feature scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
attributes = list(XData.columns)
s_scaler = StandardScaler()
XData_scaled= s_scaler.fit_transform(XData)
XData_scaled = pd.DataFrame(XData_scaled, columns=attributes)   
XData_scaled.describe().T
# Data after feature scaling


# %%
plt.figure(figsize=(20,10))
sb.boxenplot(data = XData_scaled,palette = colors)
plt.xticks(rotation=60)
plt.show()
plt.savefig("after-feature-scaling.svg")

# %% [markdown]
# ### Key Points
# 1. We can see from the above plot that the data is in the same range after the feature scaling
# 2. We can see some outliers in some attributes after the feature scaling
# 3. These outliers are not data entry errors or measurement errors as this data has been extracted from **CTG** data
# 4. Hence the outliers cannot be eliminated or dropped of from the model training as this will result in loss of information, and overfitting of the model (*fits exactly against training data.*)

# %% [markdown]
# ## Model Building (YTBD)
# 

# %% [markdown]
# ### Splitting the data into train and test datasets (yet to be done) - First Review

# %%
from sklearn.model_selection import train_test_split
XDataTrain, XDataTest, YTrain,YTest = train_test_split(XData,YData,test_size=0.3,stratify=YData)

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy="minority")
x_sam,y_sam = smote.fit_resample(XDataTrain,YTrain)
XDataTrain,YTrain = smote.fit_resample(x_sam,y_sam)
YTrain.value_counts()

# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#Training Linear Regression Classifier Model

LinearRegressionModel = Pipeline([('lr_classifier',LogisticRegression(random_state=25))])
LinearRegressionModel.fit(XDataTrain.values, YTrain.values)
filename = "./major/predict/LinearRegressionModel.sav"
joblib.dump(LinearRegressionModel,filename)
cv_score = cross_val_score(LinearRegressionModel, XDataTrain,YTrain, cv=12)
print(f"LinearRegression - {cv_score.mean()}")

# %%
linear_pred = LinearRegressionModel.predict(XDataTest)
print(classification_report(YTest,linear_pred))

# %%
DecisionTreeClassifierModel = Pipeline([('dt_classifier',DecisionTreeClassifier(random_state=25))])
DecisionTreeClassifierModel.fit(XDataTrain.values, YTrain)
filename = "./major/predict/DecisionTreeClassifierModel.sav"
joblib.dump(DecisionTreeClassifierModel,filename)
cv_score = cross_val_score(DecisionTreeClassifierModel, XDataTrain,YTrain, cv=12)
print(f"DecisionTreeClassifier - {cv_score.mean()}")

# %%
decisionTree_pred = DecisionTreeClassifierModel.predict(XDataTest)
print(classification_report(YTest,decisionTree_pred))

# %%
RandomForestClassifierModel = Pipeline([('rf_classifier',RandomForestClassifier())])
RandomForestClassifierModel.fit(XDataTrain, YTrain)
filename = "./major/predict/RandomForestClassifierModel.sav"
joblib.dump(RandomForestClassifierModel,filename)
cv_score = cross_val_score(RandomForestClassifierModel, XDataTrain,YTrain, cv=12)
print(f"RandomForestClassifier - {cv_score.mean()}")

# %%
randomForest_pred = RandomForestClassifierModel.predict(XDataTest)
print(classification_report(YTest,randomForest_pred))

# %%
SVCModel = Pipeline([('sv_classifier',SVC())])
SVCModel.fit(XDataTrain, YTrain)
filename = "./major/predict/SVCModel.sav"
joblib.dump(SVCModel,filename)
cv_score = cross_val_score(SVCModel, XDataTrain,YTrain, cv=12)
print(f"SVC - {cv_score.mean()}")

# %%
svc_pred = SVCModel.predict(XDataTest)
print(classification_report(YTest,svc_pred))

# %%
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifierModel = Pipeline([('gbcl_classifier',GradientBoostingClassifier())])
GradientBoostingClassifierModel.fit(XDataTrain, YTrain)
filename = "./major/predict/GradientBoostingClassifierModel.sav"
joblib.dump(GradientBoostingClassifierModel,filename)
cv_score = cross_val_score(GradientBoostingClassifierModel, XDataTrain,YTrain, cv=12)
print(f"GradientBoostingClassifierModel - {cv_score.mean()}")

# %%
gradient_pred = GradientBoostingClassifierModel.predict(XDataTest)
print(classification_report(YTest,gradient_pred))

# %%
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifierModel = Pipeline([('knn_classifier',KNeighborsClassifier())])
KNeighborsClassifierModel.fit(XDataTrain, YTrain)
filename = "./major/predict/KNeighborsClassifierModel.sav"
joblib.dump(KNeighborsClassifierModel,filename)
cv_score = cross_val_score(KNeighborsClassifierModel, XDataTrain,YTrain, cv=12)
print(f"KNeighborsClassifierModel - {cv_score.mean()}")

# %%
knn_pred = KNeighborsClassifierModel.predict(XDataTest)
print(classification_report(YTest,knn_pred))

# %%
GBCLModelTrain = GradientBoostingClassifierModel.score(XDataTrain,YTrain)
GBCLModelTest = GradientBoostingClassifierModel.score(XDataTest,YTest)
cv_score = cross_val_score(GradientBoostingClassifierModel, XDataTrain,YTrain, cv=12)
print(f"GradientBoostingClassifierModel - {cv_score.mean()}")

print(f"r^2(coeff of determination) on train set = {round(GBCLModelTrain, 3)}")
print(f"r^2(coeff of determination) on test set = {round(GBCLModelTest, 3)}")


RF_model = RandomForestClassifier(n_estimators= 100,criterion='entropy', max_depth=14, max_features= 'auto',  n_jobs=None)
RF_model.fit(XDataTrain, YTrain)

cv_score = cross_val_score(RF_model, XDataTrain,YTrain, cv=12)
print(f"RandomForestClassifierModelHyperTuned - {cv_score.mean()}")
RFCLModelTrain = RF_model.score(XDataTrain,YTrain)
RFCLModelTest = RF_model.score(XDataTest,YTest)
filename = "./major/predict/RandomForestClassifierModelHyperTuned.sav"
joblib.dump(RF_model,filename)

print(f"r^2(coeff of determination) on train set = {round(RFCLModelTrain, 3)}")
print(f"r^2(coeff of determination) on test set = {round(RFCLModelTest, 3)}")


# %%
pred_gbcl = RF_model.predict(XDataTest)
acc = accuracy_score(YTest,pred_gbcl)
print(f" Testing Score of the model is {acc}")

# %%
print(classification_report(YTest, pred_gbcl))

# %%
import seaborn as sns
plt.subplots(figsize=(12,8))
cf_matrix = confusion_matrix(YTest, pred_gbcl)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap=sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True),annot = True, annot_kws = {'size':20})
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


