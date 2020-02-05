import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def init():
    dataset = pd.read_csv('headbrain.csv')
    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)
    return X, Y, X_train, X_test, Y_train, Y_test

def trainModel(X_train, Y_train):
    X_mean = np.mean(X_train)
    Y_mean = np.mean(Y_train)

    numerator = np.sum((X_train - X_mean) * (Y_train - Y_mean))
    denominator = np.sum((X_train - X_mean) ** 2)

    betaOne = numerator / denominator
    betaZero = Y_mean - betaOne * X_mean

    return betaZero, betaOne

def displayPredictions(X_data, Y_data, betaZero, betaOne):
    X_min = np.min(X_data) - 100
    X_max = np.max(X_data) + 100

    X_axis =  np.linspace(X_min, X_max, 1000)
    Y_axis_prediction = betaZero + betaOne * X_axis

    plt.title('Linear Regression')

    plt.plot(X_axis, Y_axis_prediction)
    plt.scatter(X_data, Y_data, c = 'red')
    
    plt.legend(['Calculated train Value', 'Actual train Value'])
    plt.xlabel('Head Size (cm^3)')
    plt.ylabel('Brain Weight (grams)')
    plt.show()


def calculateAccuracy(X_data, Y_data, betaZero, betaOne):
    # RSS (Residual Sum of Squares) = sum of the squared residuals
    # MSE (Mean Squared Error) = mean of RSS
    # RMSE (Root Mean Square Error) = square root of MSE
    # TSS (Total Sum of Squares) is related with variance and not a metric on regression models
    Y_prediction = betaZero + betaOne * X_data
    n = len(X_data)
    RSS = np.sum((Y_data - Y_prediction) ** 2)
    MSE = RSS / n
    RMSE = np.sqrt(MSE)
    TSS = np.sum((Y_data - np.mean(Y_data)) ** 2)
    R2 = 1 - (RSS / TSS)
    
    return RMSE, R2

def main():
    X, Y, X_train, X_test, Y_train, Y_test = init()
    
    betaZero, betaOne = trainModel(X_train, Y_train)
    
    displayPredictions(X_train, Y_train, betaZero, betaOne)
    
    displayPredictions(X_test, Y_test, betaZero, betaOne)

    train_RMSE, train_R2 = calculateAccuracy(X_train, Y_train, betaZero, betaOne)
    test_RMSE, test_R2 = calculateAccuracy(X_test, Y_test, betaZero, betaOne)

    print('\nTraining Set')
    print(f'RMSE = {train_RMSE}\nR2 = {train_R2}')
    print('\nTesting Set')
    print(f'RMSE = {test_RMSE}\nR2 = {test_R2}')

if __name__ == '__main__':
    main()