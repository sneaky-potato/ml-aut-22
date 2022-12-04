import pandas as pd
import numpy as np
from Models import MLP

def read_data():
    cols_list = [
        'Target Class',
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
    
    return df

# Splitting dataset into train and test set
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

# Standard Scalar Normalization
def normalize(xTrain, xTest):

    xTrainMean = np.mean(xTrain, axis=0)
    xTrainStd = np.std(xTrain, axis=0)
    xTrainScaled = (xTrain - xTrainMean) / xTrainStd
    xTestScaled = (xTest - xTrainMean) / xTrainStd

    return xTrainScaled, xTestScaled

# Funciton for returning set of best features from forward feature selection
def forward_feature_selection(model, xTest, yTestTrue):
    base_err = 1e5
    features = []
    features_sample = [*range(0, len(model.xTrain[0]))]

    while(True):
        best_e = 1
        best_feature = None

        for i in features_sample:
            xTrainTemp = model.xTrain[:, [i]]
            xTestTemp = xTest[:, [i]]

            # Get F union x_i features
            for j in features:
                xTrainTemp = np.concatenate([xTrainTemp, model.xTrain[:, [j]]], axis=1)
                xTestTemp = np.concatenate([xTestTemp, xTest[:, [j]]], axis=1)
                
            # Training the model against current features
            mlp = MLP(model.hidden_layer_sizes, xTrainTemp, model.yTrain, model.learning_rate)
            mlp.fitModel()
            mlp.getPredict(xTestTemp)

            e = mlp.error(yTestTrue)

            if(e < best_e):
                best_e = e
                best_feature = i

        if(best_e < base_err): 
            base_err = best_e
        else:
            # In case of no better error feature is found
            best_feature = None 

        print("best feature found =>", best_feature)

        # Termination condition
        if(best_feature is None): break

        # Adding current best feature to final list
        features.append(best_feature)

        # Removing the feature from original list
        features_sample.remove(best_feature)

    print("Final accuracy after forward feature selection ->", 1 - best_e)
    return features
