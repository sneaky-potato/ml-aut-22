import pandas as pd
import numpy as np
from Models import MLP

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

def forward_feature_selection(model, xTest, yTestTrue):
    base_err = model.error(yTestTrue)
    features = []
    features_sample = [*range(0, len(model.xTrain[0]))]

    while(True):
        best_e = 1
        best_feature = None

        for i in features_sample:
            xTrainTemp = model.xTrain[:, [i]]
            xTestTemp = xTest[:, [i]]

            for j in features:
                xTrainTemp = np.concatenate([xTrainTemp, model.xTrain[:, [j]]], axis=1)
                xTestTemp = np.concatenate([xTestTemp, xTest[:, [j]]], axis=1)
                

            mlp = MLP(model.hidden_layer_sizes, xTrainTemp, model.yTrain, model.learning_rate)
            mlp.fitModel()
            mlp.getPredict(xTestTemp)

            e = mlp.error(yTestTrue)

            if(e < best_e):
                best_e = e
                best_feature = i

        if(best_e >= base_err): best_feature = None 

        print("best feature found =>", best_feature)
        if(best_feature is None): break

        features.append(best_feature)
        features_sample.remove(best_feature)

    return features
