from joblib import dump, load#joblib. dump() and joblib. load() 
#provide a replacement for pickle to work efficiently on arbitrary
 #Python objects containing large data,
import pandas as pd
from sklearn.model_selection import train_test_split
#train_test_split is a function in Sklearn model selection 
#for splitting data arrays into two subsets:
from sklearn.metrics import accuracy_score, confusion_matrix
#this function computes subset accuracy: the set of labels 
#predicted for a sample must exactly match the corresponding set of labels in y_true.
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Random forest
from sklearn.ensemble import RandomForestClassifier
#A random forest is a meta estimator that fits a number of 
#decision tree classifiers on various sub-samples of the dataset 
#and uses averaging to improve the predictive accuracy and control
 #over-fitting. The sub-sample size is controlled with the max_samples
 #parameter if bootstrap=True (default), otherwise the whole dataset 
 #is used to build each tree.
# KNN 
from sklearn.neighbors import KNeighborsClassifier
# SVM
from sklearn import svm

def load_train_dataset():
  df_train = pd.read_csv('training.csv')
  cols = df_train.columns
  cols = cols[:-1]
  train_features = df_train[cols]
  train_labels = df_train['prognosis']

  return train_features, train_labels, df_train

#load train dataset
train_features, train_labels, train_df= load_train_dataset()

# number of diseases
diseases = []

for i in train_labels:
  if i not in diseases:
    diseases.append(i)

print("Number of diseases: "+str(len(diseases)))
print('Diseases: ')
print(diseases)

#Spliting Dataset
x_train,x_val,y_train,y_val = train_test_split(train_features, train_labels,test_size=0.33, random_state=101)
#random_state is basically used for reproducing your 
#problem the same every time it is run. If you do not 
#use a random_state in train_test_split, every time 
#you make the split you might get a different set of 
#train and test data points 
print()

#Import Randomforest
RandomForest = RandomForestClassifier(n_estimators=10)
#The number of trees in the forest.

#Import KNN
KNN = KNeighborsClassifier(n_neighbors=3)

#Import SVM
SVM= svm.SVC(kernel='linear')
#Linear Kernel is used when the data is Linearly separable
 #It is mostly used when there are a Large number of Features in a particular Data Set.

def train_model(model_name):
  classifier = model_name.fit(x_train, y_train)
  #Fit function adjusts weights according to data 
  #values so that better accuracy can be achieved. 
  y_pred = classifier.predict(x_val)
  # predict() function accepts only a single argument 
  #which is usually the data to be tested.
  train_accuracy = accuracy_score(y_val,y_pred)
  #dump(classifier,"./saved_models/" + str(model_name)+ ".joblib")

  return train_accuracy

# Train and Display Training Accuracy

train_accuracy = train_model(RandomForest)
print(f'Random forest train_accuracy: {train_accuracy} ')
fig = plt.figure(figsize=(15, 10))

train_accuracy = train_model(KNN)
print(f'KNN train_accuracy: {train_accuracy} ')

train_accuracy = train_model(SVM)
print(f'SVM train_accuracy: {train_accuracy} ')

# make prediction
def make_predictions(model_name=None, test_data=None):
  classifier= load("./saved_models/" + str(model_name)+ ".joblib")
  #Save and Load Machine Learning Models in Python with scikit-learn
  if test_data is not None:
      result = classifier.predict(test_data)
      return result


preds = make_predictions('RandomForest',train_features)
cf_matrix = confusion_matrix(train_labels, preds)
ax = sns.heatmap(cf_matrix,fmt='g', annot=True, cmap='Blues')
 # If True, write the data value in each cell.
 #fmt parameter allows to add string (text) values on the cell
 # can customize the colors in your heatmap with the cmap parameter of the heatmap() 
ax.xaxis.set_ticklabels(diseases, rotation=90)
ax.yaxis.set_ticklabels(diseases,rotation=360)
ax.set_title('RandomForest')
plt.show()

# preds = make_predictions('KNN',train_features)
# cf_matrix = confusion_matrix(train_labels, preds)
# ax = sns.heatmap(cf_matrix,fmt='g', annot=True, cmap='Blues')
# ax.xaxis.set_ticklabels(diseases, rotation=90)
# ax.yaxis.set_ticklabels(diseases,rotation=360)
# ax.set_title('KNN')
# plt.show()

# preds = make_predictions('SVM',train_features)
# cf_matrix = confusion_matrix(train_labels, preds)
# ax = sns.heatmap(cf_matrix,fmt='g', annot=True, cmap='Blues')
# ax.xaxis.set_ticklabels(diseases, rotation=90)
# ax.yaxis.set_ticklabels(diseases,rotation=360)
# ax.set_title('SVM')
# plt.show()