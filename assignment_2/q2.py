from Models import BinarySVM, MLP
import numpy as np
import statistics as st
from data_processing import read_data, trainTestSplit, normalize, forward_feature_selection
import matplotlib.pyplot as plt

def main():
    x, y = read_data()

    xTrainSet, xTestSet, yTrainSet, yTestSet = trainTestSplit(x, y)
    xTrainSet, xTestSet = normalize(xTrainSet, xTestSet)

    print("Number of instances in training set => {0}, and in testing set => {1}".format(len(xTrainSet), len(xTestSet)))
    print("Feature array shape =>", xTrainSet.shape)
    print("Label array shape =>", yTrainSet.shape)
    print("Labels in dataset =>", np.unique(yTrainSet))

    print("\nModel scores:")

    print("\nSupport Vector Machine")
    bsvmList = []
    kernels = ['linear', 'poly', 'rbf']

    for kernel in kernels:
        BSVM = BinarySVM(xTrain=xTrainSet, yTrain=yTrainSet, kernel=kernel)
        BSVM.fitModel()
        yPrediction = BSVM.getPredict(xTest=xTestSet)
        accuracy = BSVM.accuracy(yTestTrue=yTestSet)

        print("::{0} SVM accuracy:: =>".format(kernel), accuracy)
        bsvmList.append((BSVM, accuracy))

    bestBSVM = max(bsvmList, key=lambda tup: tup[1])[0]
    print("Best accuracy recorded =>", bestBSVM.kernel, "SVM")

    print("\nMultilayer Perceptron")
    mplList = []
    hidden_layer_sizes=[(16,), (256, 16)]

    for hidden_layer_size in hidden_layer_sizes:
        MPLC = MLP(hidden_layer_sizes=hidden_layer_size, xTrain=xTrainSet, yTrain=yTrainSet)
        MPLC.fitModel()
        yPrediction = MPLC.getPredict(xTest=xTestSet)
        accuracy = MPLC.accuracy(yTestTrue=yTestSet)

        print("::{0} MLP accuracy:: =>".format(hidden_layer_size), accuracy)
        mplList.append((MPLC, accuracy))

    bestMLPC = max(mplList, key=lambda tup: tup[1])[0]
    print("Best accuracy recorded =>", bestMLPC.hidden_layer_sizes, "MLP")

    learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    accuracy_list =[]

    print("\nMLP with different learning rates")

    for lr in learning_rate_list:
        MPLC = MLP(hidden_layer_sizes=bestMLPC.hidden_layer_sizes, xTrain=xTrainSet, yTrain=yTrainSet, learning_rate=lr)
        MPLC.fitModel()
        yPrediction = MPLC.getPredict(xTest=xTestSet)
        accuracy = MPLC.accuracy(yTestTrue=yTestSet)

        print("::MLP with learning rate {} accuracy:: =>".format(lr), accuracy)
        accuracy_list.append(accuracy)

    plt.xlim(0, 0.15)
    plt.ylim(0.75, 1.01)
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.plot(learning_rate_list, accuracy_list, '-ok')

    plt.savefig("learning_rate_vs_accuracy.png", format="png")
    plt.show()

    print("\n::Running forward feature selection algorithm::")
    features = forward_feature_selection(bestMLPC, xTestSet, yTestSet)

    print("Set of features (indices of features) after forward feature selection ->", features)

    # Ensemble learning
    yEnsembledPred = np.zeros(yPrediction.shape)

    for y in range(len(yEnsembledPred)):
        yEnsembledPred[y] = st.mode([bestMLPC.yTest[y], bsvmList[1][0].yTest[y], bsvmList[2][0].yTest[y]])

    ensembledAccuracy = np.mean(yEnsembledPred == yTestSet)
    print("\nAccuracy after ensembling max voting =", ensembledAccuracy)

if __name__ == "__main__":
    main()

