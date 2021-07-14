import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
from matplotlib import rc
from rich import print
from matplotlib import cm


plt.rcParams["figure.figsize"] = (1920 / 100 , 902 / 100)
plt.rcParams['font.family'] = 'Times New Roman'
font = {'size'   : 35}
rc('text', usetex=True)


################
#Boosting vector
################

def vBoost(vecDim):
  vecBoost = np.zeros(shape = (vecDim,))
  vecBoost[0] = 0.9
  # vecBoost[1] = 0.1
  # vecBoost[2] = 0.1
  vecBoostMag = np.linalg.norm(vecBoost) 
  return vecBoost, vecBoostMag

################
#Boosting matrix
################

def mBoost(vecBoost, vecBoostMag , vecDim):
  matDim = vecDim+1
  matBoost = np.zeros(shape = (matDim,matDim))
  c = 1.
  if vecBoostMag == 0:
    vecBoostMag = 1
  beta = vecBoostMag / c
  gamma = 1 / np.sqrt(1 - beta**2)
  matBoost[0,0] = gamma
  print("beta = ", beta)
  print("gamma = ", gamma)
  # if vecBoostMag == 0:
    # vecBoostMag = 1
  for i in range (1, matDim):
    matBoost[i,i] = 1 + (gamma -1) * ( vecBoost[i-1] / vecBoostMag )**2 
    matBoost[0,i] = - gamma * vecBoost[i-1] 
    matBoost[i,0] = - gamma * vecBoost[i-1]
    for j in range(i+1, matDim):
      matBoost[i,j] = (gamma - 1) * (vecBoost[i-1] * vecBoost[j-1]) / (vecBoostMag**2)
      matBoost[j,i] = (gamma - 1) * (vecBoost[i-1] * vecBoost[j-1]) / (vecBoostMag**2)
  return matBoost


#####
#TSNE
#####

def vTSNE(input, d):
  xTSNE = TSNE(n_components=d).fit_transform(input)
  return xTSNE


#####
#Time-data
#####

def dtData(input, dt):
  data = input.tolist()
  data.insert(0,dt)
  data = np.array(data)
  # data = data.T
  return data

#####
#Lorentz-Invariant
#####

def lorentzInvar(input, dt):
  return dt**2 - (np.linalg.norm(input))**2


#####
# vec normalization 0 - 1
#####

def vecnormal(input):
  normalized = (input-min(input))/(max(input)-min(input))
  return normalized




#####
# rms
#####

def vecrms(input):
  rms = np.sqrt(np.square(input).sum(axis=1))
  return rms

#####
#Data preparing
#####

dfIn  = pd.read_pickle("./data/projVec.pkl")
dfTrue = pd.read_pickle("./data/true.pkl")



# print(dfIn)
# print(npDataT)
# print(type(npDataT))
# print(npDataT.shape)



# dt = 1.
vDim = 512
# vDim = 3
vB, vBmag = vBoost(vDim)

print(np.sqrt(np.square(dfIn).sum(axis=1)))


mB = mBoost(vB, vBmag, vDim)

###################################################################################################################
# Invariant Test Start
###################################################################################################################

vTestTemp = dfIn.loc[0]
# vTestTemp = vTestTemp[0:3]
lInvarIn  = lorentzInvar(vTestTemp, 0.)
# print(lInvarIn)

vInTestTemp = dtData(vTestTemp, 0)
# print(vInTestTemp)

vTestOutTemp = np.dot( vInTestTemp, mB)
# print(mB)
# print(vTestOutTemp)
# print(vTestOutTemp)
dt = vTestOutTemp[0]
# print(dt)
vTestOutTemp = np.delete(vTestOutTemp, [0])
# print(vTestOutTemp)
lInvarOut    = lorentzInvar(vTestOutTemp,dt)
# print(lInvarOut)

# print(mB)
# print(vIn)

###################################################################################################################
# Invariant Test End
###################################################################################################################



pdRest = pd.DataFrame()
pdBoost = pd.DataFrame()

# print(pdBoost)
# print(dfIn)
tRest = 0.

for i in range(0, 128):
  dfTemp = dfIn.loc[i]
  vInTemp = dtData(dfTemp, tRest)  
  dfTempNorm = vecnormal(dfTemp)
  vBmag = np.sqrt(np.square(dfTempNorm[i]))
  mB = mBoost(dfTempNorm, vBmag, vDim)
  vOutTemp = np.dot(mB, vInTemp)
  vOutTemp = pd.Series(vOutTemp)
  vInTemp = pd.Series(vInTemp)
  pdBoost = pdBoost.append(vOutTemp,ignore_index=True)
  pdRest = pdRest.append(vInTemp,ignore_index=True)

# print(pdRest)  
# print(pdBoost)  

pdBoostT = pdBoost.iloc[:,[0]]
pdBoostV = pdBoost.iloc[:, 1:]

# print(pdBoostT)
# print(pdBoostV)

pdBoostInv = lorentzInvar(pdBoostV, pdBoostT)
# print(pdBoostInv)
# print(pdBoostInv.std())

# print(pdBoost)
# print(pdRest)
# print(vOutTemp)
# # print(vIn.reshape((1,-1)))
# # print(vOut.reshape((1,-1)))

npBoost2D = vTSNE(pdBoost,2)
pdBoost2D = pd.DataFrame(npBoost2D)
npRest2D = vTSNE(pdRest,2)
pdRest2D = pd.DataFrame(npRest2D)

# print(npBoost2D)
# print(npRest2D)


# print(dfTrue)
vTrue = list(dfTrue[0])
# print(vTrue)
# print(npBoost2D)
# print(pdBoost2D)
# print(dfTrue)
# dfTrue = dfTrue[0].values
pdRest2D = pd.concat([pdRest2D,dfTrue],axis=1, ignore_index=True)
pdRest2D.columns=['x','y','c']

pdBoost2D = pd.concat([pdBoost2D,dfTrue],axis=1, ignore_index=True)
pdBoost2D.columns=['x','y','c']
# print(pdRest)

fSize = 35
nrof_labels = len(dfTrue[0])
colors = cm.rainbow(np.linspace(0, 1, nrof_labels)) 


# Scatter plot with a different color by groups

grRest2D = pdRest2D.groupby('c')
grBoost2D = pdBoost2D.groupby('c')

 
# fig, ax = plt.subplots(1,2,1)
# for name, group in groups:
#   ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
# ax.legend(fontsize=12, loc='upper left') # legend position
# plt.title('Scatter Plot of iris by matplotlib', fontsize=20)
# plt.show()


# plt.figure(0)
plt.subplot(121)
plt.title("At Rest Domain",fontsize=fSize)
for name, group in grRest2D:
  plt.plot(group.x, group.y, marker='o', linestyle='', label=name)
plt.legend(fontsize=12, loc='upper left') 
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.legend(loc='best')
plt.grid(b=True, which='both', axis='both')
plt.tight_layout()

plt.subplot(122)
plt.title("At Boosting Domain",fontsize=fSize)
for name, group in grBoost2D:
  plt.plot(group.x, group.y, marker='o', linestyle='', label=name)
plt.legend(fontsize=12, loc='upper left') 
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.legend(loc='best')
plt.grid(b=True, which='both', axis='both')
plt.tight_layout()

# plt.figure(1)
# # plt.hist(dfIn)
plt.show()
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


