import numpy as np
"""
Function to get initial cluster representatives.
"""
def getInitialRep(k,data):
    rep = []
    for i in range(k):
        rep.append(data.iloc[i])
    return rep

# function to calculate NMI
def NMI_calculator(labels, clusters):
    # calculate mutual information
    mutual_info = mutual_information(labels, clusters)
    # calculate entropy
    entropy_labels = entropy(labels)
    entropy_clusters = entropy(clusters)
    # calculate normalized mutual information
    nmi = 2 * mutual_info / (entropy_labels + entropy_clusters)
    return nmi

# function to compute mutual information
def mutual_information(labels_true, labels_pred):

    mutual_info = 0
    
    pred_classes = np.unique(labels_pred, return_counts=True)    # classes in the predicted labels with their frequency
    true_classes = len(np.unique(labels_true))                   # number of classes in the true labels

    # calculate mutual information
    for i in range(len(pred_classes[0])):
        # probability of the class in the predicted labels
        p_class = np.zeros(true_classes)
        for j in range(len(labels_pred)):

            if labels_pred[j] == pred_classes[0][i]:
                p_class[labels_true[j]-1] += 1

        p_class /= pred_classes[1][i]
        # calculate the entropy in the class i
        Entropy = 0
        for j in range(true_classes):
            if p_class[j] != 0:
                Entropy += p_class[j] * np.log2(p_class[j])
        mutual_info += pred_classes[1][i] / len(labels_true) * Entropy
    # return H(Y) - H(Y|C)
    return entropy(labels_true) + mutual_info

# function to calculate entropy
def entropy(labels):

    entropy = 0
    N = len(labels)

    # calculate probability
    prob = np.unique(labels, return_counts=True)[1] / N
    # calculate entropy
    for p in prob:
        if p != 0:
            entropy += p * np.log2(p)
    return -entropy


"""
Function for K-Means Clustering which takes in initial representatives and returns list of Jclust value over the iterations
It converges when assignment of none of the image changes i.e. in other words Jclust does not decrease and remains constant instead.
"""
def Kmeans(rep,data,y):
    N = len(data)
    maxIter = 100
    replabel = [-1]*N
    Jclust = []
    for i in range(maxIter):
        allsame = True
        # print(i)
        loss = 0
        # clustering based on current reps
        for j in range(N):
            dist = []
            for q in range(len(rep)):
                dist.append(np.linalg.norm(data.iloc[j]-rep[q]))
            if(np.argmin(dist)!=replabel[j]):
                replabel[j] = np.argmin(dist)
                allsame = False
            loss += np.min(dist)
        Jclust.append(loss/N)
        if allsame:
            NMI = NMI_calculator(replabel,y)
            print(f"For k = {len(rep)}:\nTerminating at {i-1}th iteration, Loss = {loss/N}, NMI = {NMI}")
            break
        
        # updating reps based on new clustering
        seen = np.zeros((len(rep),1))
        for j in range(N):
            rep[replabel[j]] = (rep[replabel[j]]*seen[replabel[j]] + data.iloc[j])/(seen[replabel[j]]+1)
            seen[replabel[j]] += 1;
    return Jclust, rep, NMI