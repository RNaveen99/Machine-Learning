import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def init():
    dataset = pd.read_csv('headbrain.csv')
    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)
    
    return X, Y, X_train, X_test, Y_train, Y_test

def calculateCost(theta, X_data, Y_data):
    m = len(X_data)
    prediction = X_data.dot(theta)
    cost = (1 / m) * np.sum(np.square(prediction - Y_data))
    return (cost ** (1 / 2))

def gradientDescent(X_train, X_test, Y_train, Y_test, learningRate, iterations):
    
    X_train = X_train.reshape(len(X_train), 1)
    X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis = 1)
    
    X_test = X_test.reshape(len(X_test), 1)
    X_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis = 1)
    
    m = len(Y_train)
    theta = np.zeros(X_train.shape[1]).T
    RMSE_train = np.empty(iterations)
    RMSE_test = np.empty(iterations)
    it = np.arange(iterations)

    for i in range(iterations):
        predictions = X_train.dot(theta)
        errors = predictions - Y_train
        theta = theta - (1 / m) * learningRate * (X_train.T.dot(errors))
        RMSE_train[i] = calculateCost(theta, X_train, Y_train)
        RMSE_test[i] = calculateCost(theta, X_test, Y_test)
        if i % 1000 == 0:
            plt.plot(np.arange(i), RMSE_train[0:i], color = 'red')
            plt.plot(np.arange(i), RMSE_test[0:i], color = 'blue')
            plt.xlabel('iterations')
            plt.ylabel('RMSE')
            plt.show()
        
    # plt.plot(it, RMSE_train, color = 'red')
    # plt.plot(it, RMSE_test, color = 'blue')
    # plt.xlabel('iterations')
    # plt.ylabel('RMSE')
    # plt.show()
    return theta, RMSE_train, RMSE_test

def main():
    X, Y, X_train, X_test, Y_train, Y_test = init()
    
    theta, RMSE_train, RMSE_test = gradientDescent(X_train, X_test, Y_train, Y_test, learningRate = 0.01, iterations = 5000)
    
    print(theta)
    plt.scatter(X_train, Y_train)
    predictions = theta[0] + theta[1] * X_train
    plt.plot(X_train, predictions)
    plt.show()

if __name__ == '__main__':
    main()