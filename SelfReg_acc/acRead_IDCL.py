import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from matplotlib import cm
from matplotlib import rc
import pretty_errors


plt.rcParams["figure.figsize"] = (1920 / 100 , 902 / 100)
plt.rcParams['font.family'] = 'Times New Roman'
font = {'size'   : 20}
rc('text', usetex=True)

def get_path(PATH):
  accPATH = []
  for (path, dir, files) in os.walk(PATH):
      for filename in files:
          ext = os.path.splitext(filename)[-1]
          if ext == '.pkl':
              accPATH.append("%s/%s" % (path, filename))
              # print("%s/%s" % (path, filename))
  return accPATH

PATH = './results/acc_idcl_2nd'
accs_path = get_path(PATH)

print(accs_path)
df_accs_cl = pd.DataFrame()
df_accs_cl_t = pd.DataFrame()
df_accs_self = pd.DataFrame()
df_accs_self_t = pd.DataFrame()

for idx in accs_path:
  if idx.endswith('cl.pkl'):
    # print(type(idx))
    df_temp = pd.read_pickle(idx)
    print(type(df_temp))
    df_accs_cl = df_accs_cl.append(df_temp, ignore_index=True)
    # print(df_accs)
  elif idx.endswith('cl_t.pkl'):
    # print(type(idx))
    df_temp = pd.read_pickle(idx)
    print(type(df_temp))
    df_accs_cl_t = df_accs_cl_t.append(df_temp, ignore_index=True)
  elif idx.endswith('self.pkl'):
    # print(type(idx))
    df_temp = pd.read_pickle(idx)
    print(type(df_temp))
    df_accs_self = df_accs_self.append(df_temp, ignore_index=True)
  elif idx.endswith('self_t.pkl'):
    # print(type(idx))
    df_temp = pd.read_pickle(idx)
    print(type(df_temp))
    df_accs_self_t = df_accs_self_t.append(df_temp, ignore_index=True)  
  # print(df_accs)

df_accs_cl.rename(columns = {"Accuracy":"cl"}, inplace=True)
df_accs_cl_t.rename(columns = {"Accuracy":"cl_t"}, inplace=True)
df_accs_self.rename(columns = {"Accuracy":"self"}, inplace=True)
df_accs_self_t.rename(columns = {"Accuracy":"self_t"}, inplace=True)

# print(df_accs_cl)
# print(df_accs_cl_t)
# print(df_accs_self)
# print(df_accs_self_t)

df_accs = pd.DataFrame()

df_accs = pd.concat([df_accs_cl, df_accs_cl_t, df_accs_self, df_accs_self_t], axis=1)
print(df_accs)

def meanAcc(dataIn):
  # dataIn = dataIn.Accuracy
  dataIn = pd.to_numeric(dataIn) 
  return dataIn.mean()

def stdAcc(dataIn):
  # dataIn = dataIn.Accuracy
  dataIn = pd.to_numeric(dataIn) 
  return dataIn.std()

x = ["$\mathcal{L}_{cl}$", "$\mathcal{L}_{cl,t}$","$\mathcal{L}_{SelfReg}$", "$\mathcal{L}_{SelfReg,t}$"]
y = [meanAcc(df_accs.cl), meanAcc(df_accs.cl_t),meanAcc(df_accs.self), meanAcc(df_accs.self_t)]
ystd = [stdAcc(df_accs.cl), stdAcc(df_accs.cl_t),stdAcc(df_accs.self), stdAcc(df_accs.self_t)]

print(y)
print(ystd)

lSize = 40
fSize = 25
nbins = 15



plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.title("$\mathcal{L}_{cl}$",fontsize=fSize)
plt.hist(pd.to_numeric(df_accs.cl), nbins,  color='g', alpha=0.75, edgecolor='k',range=[70, 100])
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,2)
plt.title("$\mathcal{L}_{cl,t}$",fontsize=fSize)
plt.hist(pd.to_numeric(df_accs.cl_t), nbins,  color='b', alpha=0.75, edgecolor='k',range=[70, 100])
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,3)
plt.title("$\mathcal{L}_{SelfReg}$",fontsize=fSize)
plt.hist(pd.to_numeric(df_accs.self), nbins,  color='r', alpha=0.75, edgecolor='k',range=[70, 100])
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,4)
plt.title("$\mathcal{L}_{SelfReg,t}$",fontsize=fSize)
plt.hist(pd.to_numeric(df_accs.self_t), nbins,  color='m', alpha=0.75, edgecolor='k',range=[70, 100])
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.grid(b=True, which='both', axis='both')

plt.tight_layout()

plt.figure(figsize=(10, 10))
# plt.title("Classical - without IDCL",fontsize=fSize)
plt.title("IDCL",fontsize=fSize)
plt.errorbar(x, y, yerr=ystd , linestyle="none", fmt='or' ,ecolor='k', capthick=2, capsize=5, color = 'r')
plt.xticks(fontsize=fSize)
plt.yticks(fontsize=fSize)
plt.grid(b=True, which='both', axis='both')
plt.tight_layout()

plt.show()


# print("cl")
# # print(df_cl)
# print(meanAcc(df_cl))
# print(stdAcc(df_cl))

# print("cl-lorentz")
# # print(df_cl_lorentz)
# print(meanAcc(df_cl_lorentz))
# print(stdAcc(df_cl_lorentz))

# print("self")
# # print(df_self)
# print(meanAcc(df_self))
# print(stdAcc(df_self))

# print("self-lorentz")
# # print(df_self_lorentz)
# print(meanAcc(df_self_lorentz))
# print(stdAcc(df_self_lorentz))

