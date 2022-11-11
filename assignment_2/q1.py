import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import normalize,read_data
from Kmeans import getInitialRep,Kmeans

df = read_data()

x = df.drop(columns=["Target Class"],axis=1)
y = df["Target Class"]

x_normalised,y_normalised = normalize(x,y)

pca = PCA(0.95)
principalComponents = pca.fit_transform(x_normalised)
principalDf = pd.DataFrame(data = principalComponents)

plt.ylabel("Cumulative Variance")
plt.xlabel("Number of Principal Components")
var = pca.explained_variance_ratio_
for i in range(len(var)-1):
    var[i+1] = var[i+1]+var[i]
print("Cumulative Variance based on number of components : ",var)
plt.plot(var)
plt.savefig("Variance_VS_Component_number.png", format="png")
plt.show()

# print("Scatter plot of 1st component before and after PCA: ")
plt.xlabel("1st Component After PCA")
plt.ylabel("2st Component After PCA")
plt.scatter(principalDf[principalDf.columns[0]],principalDf[principalDf.columns[1]],c=y)
plt.savefig("PCA_visulation.png", format="png")
plt.show()

K,NMI = [],[]
for k in range(2,9):
    J, reps, nmi = Kmeans(getInitialRep(k,principalDf),principalDf,y)
    K.append(k)
    NMI.append(nmi)

print(np.argmax(NMI)+2, " :is value of K for which NMI is Maximum")

print("NMI List with K from [2-8] : ", NMI)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("NMI")
plt.plot(K,NMI)
plt.savefig("K_VS_NMI.png", format="png")
plt.show()
