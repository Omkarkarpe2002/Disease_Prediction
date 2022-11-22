from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


#load test dataset
def load_test_dataset():
  df_test = pd.read_csv('testing.csv')
  cols = df_test.columns
  cols = cols[:-1]
  test_features = df_test[cols]
  test_labels = df_test['prognosis']
  return test_features, test_labels, df_test

test_features, test_labels, df_test = load_test_dataset()


# Defining Prediction function
def make_predictions(model_name=None, test_data=None):
  classifier= load("./saved_models/" + str(model_name)+ ".joblib")
  if test_data is not None:
      result = classifier.predict(test_data)
      return result
  else:
    result = classifier.predict(test_features)
    test_accuracy = accuracy_score(test_labels, result)
    return test_accuracy

# Displaying Testing Accuracy
test_accuracy = make_predictions('KNN')
print(f'Testing Accuracy KNN: {test_accuracy}')
test_accuracy = make_predictions('SVM')
print(f'Testing Accuracy SVM: {test_accuracy}')
test_accuracy = make_predictions('RandomForest')
print(f'Testing Accuracy RandomForest: {test_accuracy}')
print('\n')

#Prediction on Specific data
data = [[0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# data = np.array(data)
# data = data.reshape(1,-1)

test_accuracy= make_predictions('SVM',data)
print(f'SVM: {test_accuracy} ')

test_accuracy= make_predictions('RandomForest',data)
print(f'RandomForest: {test_accuracy} ')

test_accuracy= make_predictions('KNN',data)
print(f'KNN: {test_accuracy} ')



