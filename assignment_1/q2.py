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

def removeOutliers(dataframe, alpha, beta):
    # remove outliers
    df = dataframe
    means = {}
    std_dev = {}
    columns = df.columns
    columns = columns[:-1]
    for column in columns:
        means[column] = df[column].mean()
        std_dev[column] = df[column].std()

    df["outlier_count"] = [0 for i in range(len(df))]

    for i, row in df.iterrows():
        for column in columns:
           if(row[column] > (alpha*means[column] + beta*std_dev[column])):
            df.loc[i, "outlier_count"] += 1

    max_outlier = max(df["outlier_count"])
    df1 = df[df["outlier_count"] == max_outlier]
    dataframe = dataframe.drop(df1.index)
    dataframe = dataframe.drop('outlier_count', axis=1)

    return dataframe

def main():
    df = pd.read_csv("cleaned_data.csv")

    f = open('NaiveBayes.txt', 'w')

    print("::Final set of features formed::")
    print("::Final set of features formed::", file=f)
    print(df.columns[:-1])
    print(df.columns[:-1], file=f)

    print("\nOriginal datset size =", len(df))
    print("\nOriginal datset size =", len(df), file=f)

    df = removeOutliers(df, alpha, beta)

    print("Datset size after removing outliers=", len(df))
    print("Datset size after removing outliers=", len(df), file=f)

    train, test = trainTestSplit(df, 0.2, shuffle=True)
    
    NB = NaiveBayes(train, test, n_folds)

    scores, summary = NB.fit()

    print("\n::Naive Bayes using {} cross validation::".format(n_folds))
    print("\n::Naive Bayes using {} cross validation::".format(n_folds), file=f)

    for i in range(n_folds):
        print('Iteration({}) score = {}'.format(i+1,scores[i]))
        print('Iteration({}) score = {}'.format(i+1,scores[i]), file=f)

    train_acc = sum(scores) / len(scores)
    test_acc = NB.get_test_accuracy(test, summary)

    print("\nModel scores:")
    print("Train Accuracy: {}".format(train_acc))
    print("Test Accuracy: {}".format(test_acc))

    print("\nModel scores:", file=f)
    print("Train Accuracy: {}".format(train_acc), file=f)
    print("Test Accuracy: {}".format(test_acc), file=f)

    prior, post = NB.fit(laplace_corr=True)

    print("\n::Naive Bayes using Laplace correction::")
    print("\n::Naive Bayes using Laplace correction::", file=f)

    test_acc = NB.get_test_accuracy_laplacian(prior, post)

    print("\nModel scores:")
    print("Test Accuracy: {}".format(test_acc))
    print("\nModel scores:", file=f)
    print("Test Accuracy: {}".format(test_acc), file=f)

    f.close()

if __name__ == "__main__":
    main()