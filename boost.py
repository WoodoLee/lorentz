import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

plt.rcParams["figure.figsize"] = (1920 / 200 , 902 / 100)
plt.rcParams['font.family'] = 'Times New Roman'
font = {'size'   : 35}
rc('text', usetex=True)


inPath = "./data/projVec.pkl"
dfIn  = pd.read_pickle("./data/projVec.pkl")


# print(dfIn.loc[0])
dfInTest = dfIn.loc[0]
print(type(dfInTest))
inTest = dfInTest.tolist()
dt = 0.1
print(inTest)
inTest.insert(0, dt)
print(inTest)


for i in range (0, 127):
  dfInTest = dfIn.loc[i]
  dt = 0.1
  # dfInTime = np.insert(a, 0, dt)
  dfInTestMag = np.sqrt(np.square(dfInTest).sum())
  
  # print(dfInTestMag)


###############
#Boosting vector
###############

def vBoost(vecDim):
  vecBoost = np.zeros(shape = (vecDim,))
  vecBoost[0] = 0.1
  # vecBoost[1] = 0.2
  # vecBoost[2] = 0.3
  vecBoostMag = np.linalg.norm(vecBoost) 
  return vecBoost, vecBoostMag

###############
#Boosting matrix
###############

def mBoost(vecBoost, vecBoostMag , vecDim):
  matDim = vecDim+1
  matBoost = np.zeros(shape = (matDim,matDim))
  
  c = 1.
  beta = vecBoostMag / c
  gamma = 1 / np.sqrt(1 - beta**2)
  matBoost[0,0] = gamma
  print("beta = ", beta)
  print("gamma = ", gamma)
  for i in range (1, matDim):
    matBoost[i,i] = 1 + (gamma -1) * (vecBoost[i-1]/ vecBoostMag)**2 
    matBoost[0,i] = -gamma * vecBoost[i-1] 
    matBoost[i,0] = -gamma * vecBoost[i-1]
    for j in range(i+1, matDim):
      matBoost[i,j] = (gamma - 1) * (vecBoost[i-1] * vecBoost[j-1]) / vecBoostMag**2
      matBoost[j,i] = (gamma - 1) * (vecBoost[i-1] * vecBoost[j-1]) / vecBoostMag**2
  return matBoost


vDim = 512
vB, vBmag = vBoost(vDim)
mB = mBoost(vB, vBmag, vDim)

print(mB)



# print(gamma)

# for i in range (1,512):

# for i in range (1, mDim):
#   # print(i)
#   # print("==")
#   print("--")
#   for j in range(i+1, mDim):
#     mB[i,j] = (gamma - 1) * (vB[i-1] * vB[j-1]) / vBmag**2
#     mB[j,i] = (gamma - 1) * (vB[i-1] * vB[j-1]) / vBmag**2
#     # print(i , "," , j)
#     # print(mB[i,j])
#     # print(",")
#     # print(j)
#     # print("--")



#####
#TSNE
#####
def TSNE(input, d):
  xTSNE = TSNE(n_components=d).fit_transform(input)
  return xTSNE

# dfInTest = dfIn.loc[0]
# dfInTestMag = np.sqrt(np.square(dfInTest).sum())
# normalized_df=(dfInTest-dfInTest.min())/(dfInTest.max()-dfInTest.min())
# print(dfInTestMag)
# plt.figure(1)
# plt.hist(dfInTest)

# plt.figure(2)
# plt.hist(normalized_df)


# plt.show()


# gamma = 1 / ( 1 - v**2)

# fSize = 35

# plt.figure(1)
# for i in range (0,127):
#   plt.scatter(xTest[i][0], xTest[i][1])
# plt.xticks(fontsize=fSize)
# plt.yticks(fontsize=fSize)
# # plt.xlabel(r'Time ($\mu$s)', loc='right',  fontsize=fSize)
# # plt.ylabel(r'Entries', loc='top',  fontsize=fSize)
# # plt.legend(fontsize=30, loc='upper left')
# plt.grid(b=True, which='both', axis='both')
# plt.tight_layout()
# plt.show()

