from Models import BinarySVM, MLP
from data_processing import read_data, trainTestSplit, normalize

def main():
    x, y = read_data()

    xTrainSet, xTestSet, yTrainSet, yTestSet = trainTestSplit(x, y)
    xTrainSet, xTestSet = normalize(xTrainSet, xTestSet)

    print("SVM::")
    bsvmList = []
    kernels = ['linear', 'poly', 'rbf']

    best_accuracy = 0

    for kernel in kernels:
        BSVM = BinarySVM(xTrain=xTrainSet, yTrain=yTrainSet, kernel=kernel)
        BSVM.fitModel()
        yPrediction = BSVM.getPredict(xTest=xTestSet)
        accuracy = BSVM.accuracy(yTestTrue=yTestSet)

        print("::accuracy:: ->", accuracy)
        bsvmList.append((BSVM, accuracy))

    print("MPL")
    mplList = []
    hidden_layer_sizes=[(16,), (256, 16)]

    for hidden_layer_size in hidden_layer_sizes:
        MPLC = MLP(hidden_layer_sizes=hidden_layer_size, xTrain=xTrainSet, yTrain=yTrainSet)
        MPLC.fitModel()
        yPrediction = MPLC.getPredict(xTest=xTestSet)
        accuracy = MPLC.accuracy(yTestTrue=yTestSet)

        print("::accuracy:: ->", accuracy)
        mplList.append((MPLC, accuracy))

if __name__ == "__main__":
    main()

