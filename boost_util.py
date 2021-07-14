import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
from matplotlib import rc
from rich import print
from matplotlib import cm

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
