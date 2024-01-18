#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm
from scipy import stats


# In[4]:


mock1 = pd.read_csv(r'/home/jin/다운로드/mock_1_tinker13_threshold2_z_0.695_fullbox.txt',delim_whitespace=True)
# mock2 = pd.read_csv(r'/home/jin/다운로드/mock_2_tinker13_threshold2_z_0.695_fullbox.txt',delim_whitespace=True)
# mock3 = pd.read_csv(r'/home/jin/다운로드/mock_3_tinker13_threshold2_z_0.695_fullbox.txt',delim_whitespace=True)
header = ['x','y','z','vx','vy','vz','gal_type','halo_num_centrais','central_sfr_designation','sfr_designation','halo_num_active_satellites','halo_num_quiescent_satellite','halo_id','halo_mvir','halo_rvir']
mock1.columns = header
# mock2.columns = header
# mock3.columns = header


# In[3]:


# c0 = mock1['gal_type'] == 'centrals'
# c1 = mock1['gal_type'] == 'active_satellites'
# c2 = mock1['gal_type'] == 'quiescent_satellites'

# cen = mock1.loc[c0]
# act = mock1.loc[c1]
# qui = mock1.loc[c2]


# In[10]:


# tt = mock1[['x','y','z']]
# tt.to_csv(r'/home/jin/바탕화면/FCFC-main/mock1.txt',index=False,sep='\t')
# x = mock1['x'] 
# y = mock1['y']
# z = mock1['z']
# DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*10)
# DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*10)
# DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*10)

# RR = pd.DataFrame() ## 1. 데이터 초기화
# RR['X'] = DR_x
# RR['Y'] = DR_y
# RR['Z'] = DR_z 
# RR.to_csv(r'/home/jin/바탕화면/FCFC-main/mock1_RR.txt',index=False,sep='\t')


# In[6]:


# tt = mock2[['x','y','z']]
# tt.to_csv(r'/home/jin/바탕화면/FCFC-main/mock2.txt',index=False,sep='\t')
# x = mock2['x'] 
# y = mock2['y']
# z = mock2['z']
# DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*10)
# DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*10)
# DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*10)

# RR = pd.DataFrame() ## 1. 데이터 초기화
# RR['X'] = DR_x
# RR['Y'] = DR_y
# RR['Z'] = DR_z 
# RR.to_csv(r'/home/jin/바탕화면/FCFC-main/mock2_RR.txt',index=False,sep='\t')


# In[7]:


# tt = mock3[['x','y','z']]
# tt.to_csv(r'/home/jin/바탕화면/FCFC-main/mock3.txt',index=False,sep='\t')
# x = mock3['x'] 
# y = mock3['y']
# z = mock3['z']
# DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*10)
# DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*10)
# DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*10)

# RR = pd.DataFrame() ## 1. 데이터 초기화
# RR['X'] = DR_x
# RR['Y'] = DR_y
# RR['Z'] = DR_z 
# RR.to_csv(r'/home/jin/바탕화면/FCFC-main/mock3_RR.txt',index=False,sep='\t')


# In[13]:


# RR.to_csv(r'/home/jin/바탕화면/FCFC-main/mock1_RR.txt',index=False,sep='\t')


# In[51]:


xx = mock1[['x','y','z']]
xx


# In[ ]:


def dist(x1,x2,y1,y2,z1,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
bins = np.linspace(0,200,40)

dd_n = 0
dr_n = 0

jj = [0]
dj =1000
ff = np.zeros((len(jj),len(bins)))

for rr,mm in tqdm(enumerate(jj)):
    ll = (xx['x'] < mm +dj).__and__(xx['x'] >= mm).__and__(xx['y'] < 1000).__and__(xx['y'] >=0).__and__(xx['z'] <1000).__and__(xx['z'] >=0) 
    tt = xx.loc[ll]
    x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 
    for ii,x_i in tqdm(enumerate(x)): # range(len(x_0))
        dd = dist(x_i,x,y[ii],y,z[ii],z)
        kk = pd.cut(dd,bins)
        dd_n += kk.describe()['counts']

    DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*5)
    DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*5)
    DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*5)



    for ii,DR_xi in tqdm(enumerate(DR_x)):
        dr = dist(DR_xi,DR_x,DR_y[ii],DR_y,DR_z[ii],DR_z)
        kk = pd.cut(dr,bins)
        dr_n += kk.describe()['counts']
    nndr = (len(DR_x)*(len(DR_x)-1))
    nndd = (len(x)*(len(x)-1))
    nor = nndr/nndd
    ff[rr,] = nor*(dd_n/dr_n)

# xi = ff.mean(axis=0)
# xi    
        


# In[47]:


xx = dd_n/dr_n


# In[49]:


ss = (bins[1:] + bins[:-1])/2 
ss = np.append(0,ss)

plt.plot(ss,xx*ss*ss,'ro')


# In[45]:


# xi = ff.mean(axis=0)
# xi
ff =ff-1
xi = ff.mean(axis=0)
ss = (bins[1:] + bins[:-1])/2 
ss = np.append(0,ss)

plt.plot(bins,xi*ss*ss,'ro')


# In[83]:


xi


# In[75]:


xi


# In[ ]:


dd_n =  dd_n
dr_n =  dr_n
# x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 
ff = (dd_n/dr_n)

ff = pd.DataFrame()
ff.to_csv(r'/home/jin/바탕화면/2PCF_x_0_3000Mpc.txt',index=False,sep='\t')

# RR = pd.DataFrame() ## 1. 데이터 초기화
# RR['X'] = DR_x
# RR['Y'] = DR_y
# RR['Z'] = DR_z 
# RR.to_csv(r'/home/jin/바탕화면/FCFC-main/mock3_RR.txt',index=False,sep='\t')

# xi = ff/nor -1
# xi = xi['counts'].dropna()


# In[24]:


dr_n = 0 
x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 
DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*5)
DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*5)
DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*5)

DR_x0 = DR_x 
DR_y0 = DR_y 
DR_z0 = DR_z 

for i in tqdm(range(len(DR_x0))):
    DR_xi = DR_x0[i]
    DR_yi = DR_y0[i]
    DR_zi = DR_z0[i]

    DR_x = np.delete(DR_x ,0,0)
    DR_y  = np.delete(DR_y,0,0)
    DR_z = np.delete(DR_z,0,0)

    dr = dist(DR_xi,DR_x,DR_yi,DR_y,DR_zi,DR_z)
    kk = pd.cut(dr,bins)
    dr_n += kk.describe()



# In[27]:


x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 


# In[40]:


ff = ((dd_n/dr_n)['counts'] -1 )
ff[1]


# In[14]:


# x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 
ff = (dd_n/dr_n)
nor = (len(x)*(len(x)-1))/(((len(x)*3)*(len(x)*3-1)))


xi = ff/nor -1
xi = xi['counts'].dropna()
bins = np.linspace(0,200,40)
ss = (bins[1:] + bins[:-1])/2


# In[24]:





# In[15]:


plt.plot(bins,bins*bins*xi)


# In[ ]:




