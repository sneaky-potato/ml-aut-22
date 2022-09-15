from fileinput import filename
import pandas as pd
import numpy as np

class Node:
    def __init__(self,attribute):
        self.attribute = attribute
        self.children = {}
        self.label = None
        self.isleaf = False
    
    def addChild(self,x):
        self.children[x[0]] = x[1]
    
    def isLeaf(self):
        return self.isleaf

def getEntropy(y):
    total = len(y)
    valCnt = y.value_counts()
    ans = 0
    for i in range(len(valCnt)):
        tmp = valCnt[valCnt.index[i]]/total
        ans -= tmp*np.log(tmp)
    return ans

def getIG(x,y, attr):
    initialEntropy = getEntropy(y=y)
    total = len(x[attr])
    valCnt = x[attr].value_counts()
    finalEntropy = 0
    entropies = []
    for i in range(len(valCnt)):
        yi = y[x[attr]==valCnt.index[i]]
        entropies.append(getEntropy(yi))
    for i in range(len(valCnt)):
        tmp = valCnt[valCnt.index[i]]/total
        finalEntropy += tmp*entropies[i]
    return initialEntropy - finalEntropy

def getInfoGainList(x,y,attrs):
    infoGain = []
    for attr in attrs:
        infoGain.append(getIG(x,y,attr))
    return infoGain

def getNextNode(x,y):
    # All current Attributes
    attributes = x.columns.values
    if(len(attributes)==0):
        return None

    if(len(attributes)==1):
        valCnt = y.value_counts()
        n = Node(attribute=attributes[0])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        n.isleaf = True
        return n
    
    # find information gain of all attributes 
    InfoGainPerAttr = getInfoGainList(x,y,attributes)
    
    # find attribute with maximum information gain
    i_max = np.argmax(InfoGainPerAttr)
    
    if InfoGainPerAttr[i_max]<0.001:
        valCnt = y.value_counts()
        n = Node(attribute=attributes[i_max])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        n.isleaf = True
        return n
    
    # create a node with that as attribute
    n = Node(attribute=attributes[i_max])
    # find all dataSet seperated based on attribute with max information gain
    valCnt = x[attributes[i_max]].value_counts()
    n.label = y.value_counts().index[0]
    for i in range(len(valCnt)):
        xi = x[x[attributes[i_max]]==valCnt.index[i]]
        yi = y[x[attributes[i_max]]==valCnt.index[i]]
        xi = xi.drop(columns=attributes[i_max])
        child = getNextNode(x=xi,y=yi)
        if child is not None:
            n.addChild((valCnt.index[i],child))
    return n

def decisionTree(TrainData):
    x = TrainData.drop(columns="Segmentation")
    y = TrainData["Segmentation"]
    return getNextNode(x,y)

def predict(decisionTreeNode, x):
    if decisionTreeNode.isLeaf():
        return decisionTreeNode.label
    x_bar = x[decisionTreeNode.attribute]
    if x_bar in decisionTreeNode.children:
        newNode = decisionTreeNode.children[x_bar]
    else:
        return None
    return predict(newNode,x)

def getAccuracy(root, testSet):
    accuracy = 0
    for i in range(len(testSet)):
        sample = testSet.iloc[i]
        y_hat = predict(decisionTreeNode=root,x=sample)
        if y_hat==sample["Segmentation"]:
            accuracy+=1
    accuracy = accuracy/len(testSet)
    return accuracy

def pruneNode(root, curNode, validationSet):
    noNodesBefore = 1
    noNodesAfter = 1
    curDepth = 1
    if curNode.isLeaf():
        return noNodesBefore,noNodesAfter,1
    for child in curNode.children:
        validationSet_ = validationSet[validationSet[curNode.attribute] == child]
        if(len(validationSet_) == 0):
            continue
        noNodes = pruneNode(root=root,curNode=curNode.children[child],validationSet=validationSet_)
        noNodesBefore += noNodes[0]
        noNodesAfter += noNodes[1]
        curDepth = max(noNodes[2]+1,curDepth)
    
    initialAccuracy = getAccuracy(root, validationSet)
    valCnt = validationSet["Segmentation"].value_counts()
    prev_label = curNode.label
    curNode.label = valCnt.index[0]
    curNode.isleaf = True
    newAccuracy = getAccuracy(root, validationSet)
    if newAccuracy < initialAccuracy + 0.001:
        curNode.label = prev_label
        curNode.isleaf = False
    else:
        noNodesAfter = 1
        curDepth = 1
    return noNodesBefore,noNodesAfter,curDepth

def writeTree(root, file, depth = 0):
    str1 = "\n\n###############\nDEPTH LEVEL = " + str(depth) + "\n"
    str2 = "\nNode Attribute = " + str(root.attribute) + "\n"
    if not root.isleaf:
        str3 = "Children of this node are as follows : \n"
        for i in root.children:
            str3 += "if you get a " + str(i) + " go to this node with attribute : " + root.children[i].attribute + "\n"
    else:
        str3 = "Label of this node is : " + str(root.label)
    file.writelines(str1+str2+str3)
    
    for i in root.children:
        writeTree(root.children[i],file,depth+1)

def getAccuracyWithDepth(root, testSet, accuracyWithDepth, depth=0):
    
    for i in range(len(testSet)):
        sample = testSet.iloc[i]
        update(root, sample, accuracyWithDepth, depth)
    
def update(root, sample, accuracyWithDepth, depth):
    if sample["Segmentation"] == root.label:
        accuracyWithDepth[depth] += 1
    if not root.isleaf:
        if sample[root.attribute] in root.children:
            newNode = root.children[sample[root.attribute]]
        else:
            return 
        update(newNode, sample, accuracyWithDepth, depth=depth+1)