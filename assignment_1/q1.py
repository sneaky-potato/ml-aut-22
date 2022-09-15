from DecisionTree import decisionTree,pruneNode,getAccuracy,writeTree,getAccuracyWithDepth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
data splitting
"""
def trainTestSplit(dataSet, testRatio,shuffle=True):
    if shuffle:
        dataSet = dataSet.sample(frac=1)
    n = len(dataSet)
    n_test = int(n*(testRatio))
    testSet = dataSet[:n_test]
    trainSet = dataSet[n_test:]
    return trainSet,testSet

def main():
    dataSet = pd.read_csv("cleaned_data.csv")
    beforePruningAccuracies = []
    postPruningAccuracies = []
    bestTree = None
    bestAccuracy = 0
    bestTreeDepth = None
    for i in range(10):
        print(f"\n\nFor {i}th random split: \n")
        trainSet, testSet = trainTestSplit(dataSet=dataSet,testRatio=0.2)
        trainSet, validationSet = trainTestSplit(dataSet=trainSet, testRatio=0.3)
        decisionTreeRoot = decisionTree(trainSet)
        if decisionTreeRoot is None:
            print("failed to Train")
            return
        TrainAccuracy = getAccuracy(decisionTreeRoot,testSet=trainSet)
        ValidationAccuracy = getAccuracy(decisionTreeRoot,testSet=validationSet)
        TestAccuracy = getAccuracy(decisionTreeRoot,testSet=testSet)
        print("Before Pruning Accuracy TrainSet: ", TrainAccuracy)
        print("Before Pruning Accuracy ValidationSet: ", ValidationAccuracy)
        print("Before Pruning Accuracy TestSet: ", TestAccuracy)
        beforePruningAccuracies.append(TestAccuracy)
        nodes = pruneNode(decisionTreeRoot,decisionTreeRoot,validationSet=validationSet)
        TrainAccuracy = getAccuracy(decisionTreeRoot,testSet=trainSet)
        ValidationAccuracy = getAccuracy(decisionTreeRoot,testSet=validationSet)
        TestAccuracy = getAccuracy(decisionTreeRoot,testSet=testSet)
        print("After Pruning Accuracy TrainSet: ", TrainAccuracy)
        print("After Pruning Accuracy ValidationSet: ", ValidationAccuracy)
        print("After Pruning Accuracy TestSet: ", TestAccuracy)
        postPruningAccuracies.append(TestAccuracy)
        if(bestTree==None):
            bestAccuracy = TestAccuracy
            bestTree = decisionTreeRoot
            bestTreeDepth = nodes[2]
        elif (bestAccuracy<TestAccuracy):
            bestAccuracy = TestAccuracy
            bestTree = decisionTreeRoot
            bestTreeDepth = nodes[2]
        print("No of Nodes before and after pruning:",nodes[0],nodes[1])
        print("depth of this tree is : ",nodes[2])
    
    print("-------------------------------------------\n\nBest Accuracy : ",np.max(bestAccuracy))
    trainSet, testSet = trainTestSplit(dataSet=dataSet,testRatio=0.2)
    file1 = open("DecisionTree.txt", 'w')
    if file1 is None:
        print("could not write..error!!")
    writeTree(decisionTreeRoot,file1)
    accuracyWithDepth = [0]*bestTreeDepth
    getAccuracyWithDepth(decisionTreeRoot,testSet,accuracyWithDepth)
    accuracyWithDepthPlot = [acc/len(testSet) for acc in accuracyWithDepth]
    print(accuracyWithDepthPlot)
    plt.plot(accuracyWithDepthPlot)
    plt.savefig("accuracy_VS_depth.png", format="png")
    plt.show()

if __name__ == "__main__":
    main()