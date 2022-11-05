import math
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class BinarySVM:
    def __init__(self, xTrain, yTrain, kernel) -> None:
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.kernel = kernel
        self.yTest = None
        self.__model = None
    
    def fitModel(self, degree=2):
        if(self.kernel == 'poly'):
            self.__model = SVC(kernel=self.kernel, gamma=1, degree=degree)
        else:
            self.__model = SVC(kernel=self.kernel, gamma=1)

        self.__model.fit(self.xTrain, self.yTrain)
    
    def getPredict(self, xTest):
        if(self.__model == None): print("Train the model first")

        self.yTest = self.__model.predict(xTest)
        return self.yTest

    def accuracy(self, yTestTrue):
        if(self.__model == None): print("Train the model first")

        return np.mean(yTestTrue == self.yTest)
    
class MLP:
    def __init__(self, hidden_layer_sizes, xTrain, yTrain, learning_rate=1e-3, solver='sgd', batch_size=32) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = None
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
        self.__model = None

    def fitModel(self):
        self.__model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, learning_rate_init=self.learning_rate, solver=self.solver, batch_size=self.batch_size, max_iter=500)
        self.__model.fit(self.xTrain, self.yTrain)

    def getPredict(self, xTest):
        if(self.__model == None): print("Train the model first")

        self.yTest = self.__model.predict(xTest)
        return self.yTest
    
    def accuracy(self, yTestTrue):
        if(self.__model == None): print("Train the model first")

        return np.mean(yTestTrue == self.yTest)