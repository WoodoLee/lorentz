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

PATH = './results/acc_classic_2'
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

print(df_accs_cl)
print(df_accs_cl_t)
print(df_accs_self)
print(df_accs_self_t)


accs_cl_sketch = pd.to_numeric(df_accs_cl.iloc[:,0])
accs_cl_sketch = accs_cl_sketch.dropna()
accs_cl_catoon = pd.to_numeric(df_accs_cl.iloc[:,1])
accs_cl_catoon = accs_cl_catoon.dropna()
accs_cl_painting = pd.to_numeric(df_accs_cl.iloc[:,2])
accs_cl_painting = accs_cl_painting.dropna()
accs_cl_photo = pd.to_numeric(df_accs_cl.iloc[:,3])
accs_cl_photo = accs_cl_photo.dropna()


accs_cl_t_sketch = pd.to_numeric(df_accs_cl_t.iloc[:,0])
accs_cl_t_sketch = accs_cl_t_sketch.dropna()
accs_cl_t_catoon = pd.to_numeric(df_accs_cl_t.iloc[:,1])
accs_cl_t_catoon = accs_cl_t_catoon.dropna()
accs_cl_t_painting = pd.to_numeric(df_accs_cl_t.iloc[:,2])
accs_cl_t_painting = accs_cl_t_painting.dropna()
accs_cl_t_photo = pd.to_numeric(df_accs_cl_t.iloc[:,3])
accs_cl_t_photo = accs_cl_t_photo.dropna()




accs_self_sketch = pd.to_numeric(df_accs_self.iloc[:,0])
accs_self_sketch = accs_self_sketch.dropna()
accs_self_catoon = pd.to_numeric(df_accs_self.iloc[:,1])
accs_self_catoon = accs_self_catoon.dropna()
accs_self_painting = pd.to_numeric(df_accs_self.iloc[:,2])
accs_self_painting = accs_self_painting.dropna()
accs_self_photo = pd.to_numeric(df_accs_self.iloc[:,3])
accs_self_photo = accs_self_photo.dropna()


accs_self_t_sketch = pd.to_numeric(df_accs_self_t.iloc[:,0])
accs_self_t_sketch = accs_self_t_sketch.dropna()
accs_self_t_catoon = pd.to_numeric(df_accs_self_t.iloc[:,1])
accs_self_t_catoon = accs_self_t_catoon.dropna()
accs_self_t_painting = pd.to_numeric(df_accs_self_t.iloc[:,2])
accs_self_t_painting = accs_self_t_painting.dropna()
accs_self_t_photo = pd.to_numeric(df_accs_self_t.iloc[:,3])
accs_self_t_photo = accs_self_t_photo.dropna()


# df_accs = pd.DataFrame()

# df_accs = pd.concat([df_accs_cl, df_accs_cl_t, df_accs_self, df_accs_self_t], axis=1)
# # print(df_accs)

# def meanAcc(dataIn):
#   # dataIn = dataIn.Accuracy
#   dataIn = pd.to_numeric(dataIn) 
#   return dataIn.mean()

# def stdAcc(dataIn):
#   # dataIn = dataIn.Accuracy
#   dataIn = pd.to_numeric(dataIn) 
#   return dataIn.std()

# 0
x_photo    = ["$\mathcal{L}_{cl}^{photo}$", "$\mathcal{L}_{cl,t}^{photo}$","$\mathcal{L}_{SelfReg}$^{photo}", "$\mathcal{L}_{SelfReg,t}^{photo}$"]
# 1
x_painting = ["$\mathcal{L}_{cl}^{painting}$", "$\mathcal{L}_{cl,t}^{painting}$","$\mathcal{L}_{SelfReg}$^{painting}", "$\mathcal{L}_{SelfReg,t}^{painting}$"]
# 2
x_cartoon = ["$\mathcal{L}_{cl}^{cartoon}$", "$\mathcal{L}_{cl,t}^{cartoon}$","$\mathcal{L}_{SelfReg}$^{cartoon}", "$\mathcal{L}_{SelfReg,t}^{cartoon}$"]
# 3
x_sketch = ["$\mathcal{L}_{cl}^{sketch}$", "$\mathcal{L}_{cl,t}^{sketch}$","$\mathcal{L}_{SelfReg}$^{sketch}", "$\mathcal{L}_{SelfReg,t}^{sketch}$"]




# y = [meanAcc(df_accs.cl), meanAcc(df_accs.cl_t),meanAcc(df_accs.self), meanAcc(df_accs.self_t)]
# ystd = [stdAcc(df_accs.cl), stdAcc(df_accs.cl_t),stdAcc(df_accs.self), stdAcc(df_accs.self_t)]

# print(y)
# print(ystd)

lSize = 15
fSize = 25
nbins = 30

print(accs_self_t_catoon)
def histMax(inHist):
  counts, bin_edges = np.histogram(inHist, bins=nbins,range=[50, 100])
  cMax = counts.max()
  return cMax, 

plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.title("$\mathcal{L}_{SelfReg}$",fontsize=fSize)
plt.hist(accs_self_sketch, nbins,  color='g', alpha=0.75  ,range=[50, 100], label = 'sketch')
plt.hist(accs_self_catoon, nbins,  color='b', alpha=0.75  ,range=[50, 100], label = 'catoon')
plt.hist(accs_self_painting, nbins,  color='y', alpha=0.75,range=[50, 100], label = 'painting')
plt.hist(accs_self_photo, nbins,  color='r', alpha=0.75   ,range=[50, 100], label = 'photo')

plt.errorbar(accs_self_sketch.mean(), histMax(accs_self_sketch), xerr=accs_self_sketch.std() , linestyle="none", fmt='og' ,ecolor='k', capthick=1, capsize=5, color = 'g')
plt.errorbar(accs_self_catoon.mean(), histMax(accs_self_catoon), xerr=accs_self_catoon.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'b')
plt.errorbar(accs_self_painting.mean(), histMax(accs_self_painting), xerr=accs_self_painting.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'y')
plt.errorbar(accs_self_photo.mean(), histMax(accs_self_photo), xerr=accs_self_painting.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'r')


plt.xticks(fontsize=fSize)
plt.yticks(np.arange(0, 22, 2),fontsize=fSize)
plt.legend(loc='upper left',fontsize=lSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,2)
plt.title("$\mathcal{L}_{SelfReg,t}$",fontsize=fSize)
plt.hist(accs_self_t_sketch, nbins,  color='g', alpha=0.75  , range=[50, 100],label = 'sketch')
plt.hist(accs_self_t_catoon, nbins,  color='b', alpha=0.75  , range=[50, 100],label = 'catoon')
plt.hist(accs_self_t_painting, nbins,  color='y', alpha=0.75, range=[50, 100],label = 'painting')
plt.hist(accs_self_t_photo, nbins,  color='r', alpha=0.75   , range=[50, 100],label = 'photo')


plt.errorbar(accs_self_t_sketch.mean(), histMax(accs_self_t_sketch) , xerr=accs_self_t_sketch.std() , linestyle="none", fmt='og' ,ecolor='k', capthick=1, capsize=5, color = 'g')
plt.errorbar(accs_self_t_catoon.mean(), histMax(accs_self_t_catoon) , xerr=accs_self_t_catoon.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'b')
plt.errorbar(accs_self_t_painting.mean(),histMax(accs_self_t_painting) , xerr=accs_self_t_painting.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'y')
plt.errorbar(accs_self_t_photo.mean(), histMax(accs_self_t_photo), xerr=accs_self_t_photo.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'r')


plt.xticks(fontsize=fSize)
plt.yticks(np.arange(0, 22, 2),fontsize=fSize)
plt.legend(loc='upper left',fontsize=lSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,3)
plt.title("$\mathcal{L}_{cl}$",fontsize=fSize)
plt.hist(accs_cl_sketch, nbins,  color='g', alpha=0.75  ,range=[50, 100], label = 'sketch')
plt.hist(accs_cl_catoon, nbins,  color='b', alpha=0.75  ,range=[50, 100], label = 'catoon')
plt.hist(accs_cl_painting, nbins,  color='y', alpha=0.75,range=[50, 100], label = 'painting')
plt.hist(accs_cl_photo, nbins,  color='r', alpha=0.75   ,range=[50, 100], label = 'photo')

plt.errorbar(accs_cl_sketch.mean(),   histMax(accs_cl_sketch),   xerr=accs_cl_sketch.std() , linestyle="none", fmt='og' ,ecolor='k', capthick=1, capsize=5, color = 'g')
plt.errorbar(accs_cl_catoon.mean(),   histMax(accs_cl_catoon),   xerr=accs_cl_catoon.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'b')
plt.errorbar(accs_cl_painting.mean(), histMax(accs_cl_painting), xerr=accs_cl_painting.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'y')
plt.errorbar(accs_cl_photo.mean(),    histMax(accs_cl_photo),    xerr=accs_cl_photo.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'r')


plt.xticks(fontsize=fSize)
plt.yticks(np.arange(0, 22, 2),fontsize=fSize)
plt.legend(loc='upper left',fontsize=lSize)
plt.grid(b=True, which='both', axis='both')

plt.subplot(2,2,4)
plt.title("$\mathcal{L}_{cl,t}$",fontsize=fSize)
plt.hist(accs_cl_t_sketch, nbins,  color='g', alpha=0.75  ,range=[50, 100], label = 'sketch')
plt.hist(accs_cl_t_catoon, nbins,  color='b', alpha=0.75  ,range=[50, 100], label = 'catoon')
plt.hist(accs_cl_t_painting, nbins,  color='y', alpha=0.75,range=[50, 100], label = 'painting')
plt.hist(accs_cl_t_photo, nbins,  color='r', alpha=0.75   ,range=[50, 100], label = 'photo')

plt.errorbar(accs_cl_t_sketch.mean(),   histMax(accs_cl_t_sketch),   xerr=accs_cl_t_sketch.std()   , linestyle="none", fmt='og' ,ecolor='k', capthick=1, capsize=5, color = 'g')
plt.errorbar(accs_cl_t_catoon.mean(),   histMax(accs_cl_t_catoon),   xerr=accs_cl_t_catoon.std()   , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'b')
plt.errorbar(accs_cl_t_painting.mean(), histMax(accs_cl_t_painting), xerr=accs_cl_t_painting.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'y')
plt.errorbar(accs_cl_t_photo.mean(),    histMax(accs_cl_t_photo),    xerr=accs_cl_t_photo.std() , linestyle="none", fmt='ob' ,ecolor='k', capthick=1, capsize=5, color = 'r')

plt.xticks(fontsize=fSize)
plt.yticks(np.arange(0, 22, 2),fontsize=fSize)
plt.legend(loc='upper left',fontsize=lSize)
plt.grid(b=True, which='both', axis='both')


print("accs_cl_sketch = "  , accs_cl_sketch.mean(),   accs_cl_sketch.std()  )
print("accs_cl_catoon = "  , accs_cl_catoon.mean(),   accs_cl_catoon.std()  )
print("accs_cl_paintig = " , accs_cl_painting.mean(), accs_cl_painting.std())
print("accs_cl_photo = "   , accs_cl_photo.mean(),    accs_cl_photo.std())


print("accs_cl_t_sketch = "  , accs_cl_t_sketch.mean(),   accs_cl_t_sketch.std()  )
print("accs_cl_t_catoon = "  , accs_cl_t_catoon.mean(),   accs_cl_t_catoon.std()  )
print("accs_cl_t_paintig = " , accs_cl_t_painting.mean(), accs_cl_t_painting.std())
print("accs_cl_t_photo = "   , accs_cl_t_photo.mean(),    accs_cl_t_photo.std())


print("accs_self_sketch = "  , accs_self_sketch.mean(),   accs_self_sketch.std()  )
print("accs_self_catoon = "  , accs_self_catoon.mean(),   accs_self_catoon.std()  )
print("accs_self_paintig = " , accs_self_painting.mean(), accs_self_painting.std())
print("accs_self_photo = "   , accs_self_photo.mean(),    accs_self_photo.std())


print("accs_self_t_sketch = "  , accs_self_t_sketch.mean(),   accs_self_t_sketch.std()  )
print("accs_self_t_catoon = "  , accs_self_t_catoon.mean(),   accs_self_t_catoon.std()  )
print("accs_self_t_paintig = " , accs_self_t_painting.mean(), accs_self_t_painting.std())
print("accs_self_t_photo = "   , accs_self_t_photo.mean(),    accs_self_t_photo.std())



print("avg_cl = " , (accs_cl_sketch.mean() + accs_cl_catoon.mean() + accs_cl_painting.mean() + accs_cl_photo.mean())/4          , np.sqrt(accs_cl_sketch.std()**2 + accs_cl_catoon.std()**2 + accs_cl_painting.std()**2 + accs_cl_photo.std()**2))
print("avg_cl_t = " , (accs_cl_t_sketch.mean() + accs_cl_t_catoon.mean() + accs_cl_t_painting.mean() + accs_cl_t_photo.mean())/4, np.sqrt(accs_cl_t_sketch.std()**2 + accs_cl_t_catoon.std()**2 + accs_cl_t_painting.std()**2 + accs_cl_t_photo.std()**2))
print("avg_self = " , (accs_self_sketch.mean() + accs_self_catoon.mean() + accs_self_painting.mean() + accs_self_photo.mean())/4, np.sqrt(accs_self_sketch.std()**2 + accs_self_catoon.std()**2 + accs_self_painting.std()**2 + accs_self_photo.std()**2))
print("avg_self_t = " , (accs_self_t_sketch.mean() + accs_self_t_catoon.mean() + accs_self_t_painting.mean() + accs_self_t_photo.mean())/4, np.sqrt(accs_self_t_sketch.std()**2 + accs_self_t_catoon.std()**2 + accs_self_t_painting.std()**2 + accs_self_t_photo.std()**2))

# plt.subplot(2,2,3)
# plt.title("$\mathcal{L}_{SelfReg}$",fontsize=fSize)
# plt.hist(pd.to_numeric(df_accs.self), nbins,  color='r', alpha=0.75, edgecolor='k',range=[70, 100])
# plt.xticks(fontsize=fSize)
# plt.yticks(fontsize=fSize)
# plt.grid(b=True, which='both', axis='both')

# plt.subplot(2,2,4)
# plt.title("$\mathcal{L}_{SelfReg,t}$",fontsize=fSize)
# plt.hist(pd.to_numeric(df_accs.self_t), nbins,  color='m', alpha=0.75, edgecolor='k',range=[70, 100])
# plt.xticks(fontsize=fSize)
# plt.yticks(fontsize=fSize)
# plt.grid(b=True, which='both', axis='both')

plt.tight_layout()

# plt.figure(figsize=(10, 10))
# plt.title("Classical - without IDCL",fontsize=fSize)
# # plt.title("IDCL",fontsize=fSize)
# plt.errorbar(x, y, yerr=ystd , linestyle="none", fmt='or' ,ecolor='k', capthick=2, capsize=5, color = 'r')
# plt.xticks(fontsize=fSize)
# plt.yticks(fontsize=fSize)
# plt.grid(b=True, which='both', axis='both')
# plt.tight_layout()

plt.show()

