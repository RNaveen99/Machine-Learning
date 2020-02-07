import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

def init():
    dataset = pd.read_csv('breast_cancer.csv')

    dataset = dataset.drop(['id', 'Unnamed: 32'], axis = 1)
    dataset = dataset.replace('M', 1)
    dataset = dataset.replace('B', 0)
    
    X = dataset.iloc[0:, 1:]
    Y = dataset.iloc[0: , 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

    return X_train, X_test, Y_train, Y_test
    
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def calculateCost(theta, X_data, Y_data):
    h = sigmoid(X_data.dot(theta))
    return -(Y_data.T * (np.log(h)) + (1 - Y_data.T) * (np.log(1 - h))).mean()

def logisticRegression(X_train, Y_train, learningRate, iteration):
    theta = np.zeros(X_train.shape[1])
    cost_history = []
    for i in range(iteration):
        gradient = learningRate * (np.dot(X_train.T, (sigmoid(np.dot(X_train, theta))-Y_train)) / X_train.shape[0])
        theta -= gradient
        cost = calculateCost(theta, X_train, Y_train)
        cost_history.append(cost)
    return theta, cost_history

def main():
    X_train, X_test, Y_train, Y_test = init()
    theta, cost_history = logisticRegression(X_train, Y_train, 0.01, 1000)
    prediction = sigmoid(X_test.dot(theta))

    prediction = (prediction > 0.5).astype(int)
    print(f'Accuracy is {metrics.accuracy_score(Y_test, prediction) * 100}')
    print(prediction)

if __name__ == '__main__':
    main()