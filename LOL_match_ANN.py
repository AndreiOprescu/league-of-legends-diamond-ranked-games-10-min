import pandas as pd
import numpy as np

# Opening the dataset
dataset = pd.read_csv("high_diamond_ranked_10min.csv")
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1:2].values

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Train test splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Making the model and training it
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = "relu", input_dim = X.shape[1]))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 20, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 10, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 32, epochs = 50)

# testing the model
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_model(optimizer, node_amount, dropout_rate):
    classifier = Sequential()
    classifier.add(Dense(units = node_amount[0], kernel_initializer = "uniform", activation = "relu", input_dim = X.shape[1]))
    classifier.add(Dropout(rate = dropout_rate))
    
    classifier.add(Dense(units = node_amount[1], kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dropout(rate = dropout_rate))
    
    classifier.add(Dense(units = node_amount[2], kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dropout(rate = dropout_rate))
    
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    return classifier

classifier = KerasClassifier(build_fn = build_model)
parameters = {'epochs': [25, 50], 
              'batch_size': [16, 32, 64], 
              'optimizer': ['adam', 'RMSprop'], 
              'node_amount': [[30, 20, 10], [20, 15, 10]], 
              'dropout_rate': [0.1, 0.2]}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy')
grid_search = grid_search.fit(X_train, y_train)

