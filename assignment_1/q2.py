import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayes

alpha = 1
beta = 3
n_folds = 10

def trainTestSplit(dataSet, testRatio, shuffle=True):
    if shuffle:
        dataSet = dataSet.sample(frac=1)
    n = len(dataSet)
    n_test = int(n*(testRatio))
    testSet = dataSet[:n_test]
    trainSet = dataSet[n_test:]
    return trainSet,testSet


def main():
    df = pd.read_csv("cleaned_data.csv")
    train, test = trainTestSplit(df, 0.2, shuffle=True)
    
    NB = NaiveBayes(train, test, n_folds)

    scores, summary = NB.fit()

    print("::Naive Bayes using {} cross validation::".format(n_folds))
    for i in range(n_folds):
        print('Iteration({}) score = {}'.format(i+1,scores[i]))

    train_acc = sum(scores) / len(scores)
    test_acc = NB.get_test_accuracy(test, summary)

    print("\nModel scores:")
    print("Train Accuracy: {}".format(train_acc))
    print("Test Accuracy: {}".format(test_acc))


    prior, post = NB.fit(laplace_corr=True)

    print("::Naive Bayes using Laplace correction::")

    test_acc = NB.get_test_accuracy_laplacian(prior, post)

    print("\nModel scores:")
    print("Test Accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()