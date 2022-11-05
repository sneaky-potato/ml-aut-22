import pandas as pd
import numpy as np

def read_data():
    cols_list = [
        'wine',
        'alcohol',
        'malic acid',
        'ash',
        'alcalinity of ash',
        'magnesium',
        'total phenols',
        'flavanoids',
        'nonflavanoid phenols',
        'proanthocyanins',
        'color intensity',
        'hue',
        'OD280/OD315 of diluted wines',
        'proline'
    ]

    df = pd.read_csv('wine.data', sep=',', header=None, names=cols_list)

    print(df.head())

    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return x, y

def trainTestSplit(x, y, testRatio=0.2, shuffle=True):

    if shuffle:
        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]

    spiltIndex = int((1-testRatio)*len(x) + 0.5)

    xTrainSet = x[:spiltIndex]
    yTrainSet = y[:spiltIndex]

    xTestSet = x[spiltIndex:]
    yTestSet = y[spiltIndex:]

    return xTrainSet, xTestSet, yTrainSet, yTestSet

def normalize(xTrain, xTest):

    xTrainMean = np.mean(xTrain, axis=0)
    xTrainStd = np.std(xTrain, axis=0)
    xTrainScaled = (xTrain - xTrainMean) / xTrainStd
    xTestScaled = (xTest - xTrainMean) / xTrainStd

    return xTrainScaled, xTestScaled