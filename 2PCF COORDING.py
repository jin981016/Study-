#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib.gridspec import GridSpec
from tqdm.notebook import tqdm
from scipy import stats


mock1 = pd.read_csv(r'/home/jin/다운로드/mock_1_tinker13_threshold2_z_0.695_fullbox.txt',delim_whitespace=True)
header = ['x','y','z','vx','vy','vz','gal_type','halo_num_centrais','central_sfr_designation','sfr_designation','halo_num_active_satellites','halo_num_quiescent_satellite','halo_id','halo_mvir','halo_rvir']
mock1.columns = header



def dist(x1,x2,y1,y2,z1,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
bins = np.linspace(0,200,40)

dd_n = 0
dr_n = 0

jj = [0]
dj =1000
xi = 0
ff = 0

for rr,mm in tqdm(enumerate(jj)):
    ll = (xx['x'] < mm +dj).__and__(xx['x'] >= mm).__and__(xx['y'] < 1000).__and__(xx['y'] >=0).__and__(xx['z'] <1000).__and__(xx['z'] >=0) 
    tt = xx.loc[ll]
    x ,y ,z = tt['x'].to_numpy() , tt['y'].to_numpy() , tt['z'].to_numpy() 
    for ii in tqdm(range(len(x))): # range(len(x_0))
        dd = dist(x[ii],x[ii+1:],y[ii],y[ii+1:],z[ii],z[ii+1:])
        kk = pd.cut(dd,bins)
        dd_n += kk.describe()['counts']

    DR_x = stats.uniform(x.min(),x.max()).rvs(len(x)*5)
    DR_y = stats.uniform(y.min(),y.max()).rvs(len(y)*5)
    DR_z = stats.uniform(z.min(),z.max()).rvs(len(z)*5)



    for ii in tqdm(range(len(DR_x))):
        dr = dist(DR_x[ii],DR_x[ii+1:],DR_y[ii],DR_y[ii+1:],DR_z[ii],DR_z[ii+1:])
        kk = pd.cut(dr,bins)
        dr_n += kk.describe()['counts']
    nndr = (len(DR_x)*(len(DR_x)-1))
    nndd = (len(x)*(len(x)-1))
    nor = nndr/nndd
    ff = nor*(dd_n/dr_n) -1
    xi += ff
    
    
ss = (bins[1:] + bins[:-1])/2 
ss = np.append(ss,200)
plt.plot(ss,xi*ss*ss,'ro')


