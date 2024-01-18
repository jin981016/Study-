# plotfitter_emcee_multiprocess
# Minsu Kim @ Sejong University

import glob
import multiprocessing
import os
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

matplotlib.use('agg') # DO NOT REMOVE THIS

# warnings.filterwarnings("ignore")

from class_Plotfitter_emcee import Plotfit

# #======================================================================================================================================================
# commdir = '/home/mskim/workspace/research/THINGS_RAW'
# commdir = '/home/mskim/workspace/research/Little_Things_RAW'
# commdir = '/home/mskim/workspace/research/test_superprofile/NGC2403/INCLRES/I40'
# commdir = '/home/mskim/workspace/research/comparing_LT_TH/THINGS'
# commdir = '/home/mskim/workspace/research/comparing_LT_TH/Little_Things'

commdir = '/home/mandu/workspace/research/AVID/final_cubes'

# frac = 1.0

# MOD, IANJA===================================================================================
# key = '_SP'
# fit_range = None
# fix_center = False
# fit_baseline = 1
# addtxt = ''
#==============================================================================================

#STANDARD======================================================================================
key = '' #start with _
fit_range = None
fix_center = True
fit_baseline = True
addtxt = '' #start with _
#==============================================================================================

filenames = natsorted(np.array(glob.glob(commdir+"/*/stacked_total{}.txt".format(key))))

savename_plot = commdir+"/png/{{}}_stacked{}{}_GFIT.png".format(key, addtxt)
savename_info   = commdir+"/info_stacked{}{}_GFIT.csv".format(key, addtxt)

gausss = [1,2]
# gausss = [2,3]

num_cores = 16

writenew = True

#======================================================================================================================================================

print(filenames[0])
print(savename_info)
print('FIT OPTIONS:')
print('    Fit range : {}'.format(fit_range))
print('    Fix center: {}'.format(fix_center))

seed = str(time.time()).split(".")[-1]

def task_multirun(filenames):
    
    filenames = np.array([filenames], dtype=object)

    for filename in filenames:

        x,y,err = np.genfromtxt(filename, unpack=True, usecols=(0,1,2))
        if(np.all(y==0.0)):
            f = open(os.path.dirname(filename)+"_plotfitter_temp_{}.txt".format(seed), 'w')
            f.write('NaN')
            f.close()
            return
    
        a = Plotfit()
        a.open(filename)
        a.fitgauss(gausss, pbar=False, savechain=False, fit_range=fit_range, fit_baseline=fit_baseline, fix_center=fix_center)
        galname = filename.split("/")[-2]
        a.saveplot(savename_plot.format(galname))
        # a.saveplot(savename_plot)
        df = a.getdata()
        df.to_string(os.path.dirname(filename)+"_plotfitter_temp_{}.txt".format(seed), index=False)

#WRITE NEW FILE IF THERE ISN'T ONE
def writenewfile():
    header = np.array(['GALAXY'])
    header = np.append(header, ['SN{}'.format(g) for g in gausss])
    header = np.append(header, ['N{}'.format(g) for g in gausss])
    params = ['A{}{}', 'A{}{}E+', 'A{}{}E-',
                'V{}{}', 'V{}{}E+', 'V{}{}E-',
                'S{}{}', 'S{}{}E+', 'S{}{}E-']
    params_base = ['B{}', 'B{}E+', 'B{}E-']
    for g in gausss:
        if(fit_baseline==True):
            for param in params_base:
                header = np.append(header, param.format(g))
        for i in range(g):
            for param in params:
                header = np.append(header, param.format(g, i+1))        

    galaxies = np.zeros(len(filenames), dtype=object)
    for i, filename in enumerate(filenames):
        galaxies[i] = filename.split("/")[-2]

    df = pd.DataFrame(0, index=range(len(filenames)), columns=header)
    df['GALAXY'] = galaxies
    
    return df

if(writenew==True):
    df = writenewfile()
    print(df)

else:
    if(os.path.isfile(savename_info)==False):
        print("Previous file is not detected: Writing new file")
        df = writenewfile()
    else:
        df = pd.read_csv(savename_info, delim_whitespace=True)
        gausss_orig = [col for col in df.columns if 'SN' in col]
        gausss_orig = np.array([i.split('SN')[1] for i in gausss_orig], dtype=int)
        if(np.array_equal(gausss, gausss_orig)==False):
            print("MISMATCH OF DIMENSION: Writing new file.")
            print("  Previous: {}".format(gausss_orig))
            print("   Ordered: {}".format(gausss))
            df = writenewfile()
        else:
            pass


if(len(filenames)<num_cores):
    num_cores = len(filenames)

toremoves = glob.glob(commdir+"/*_plotfitter_temp_{}.txt".format(seed))
for toremove in toremoves:
    os.remove(toremove)

lists = np.array(filenames, dtype=object)
pool = multiprocessing.Pool(processes=num_cores)

with tqdm(total=len(lists)) as pbar:
    for _ in tqdm(pool.imap_unordered(task_multirun, lists)):
        pbar.update()

pool.close()
pool.join()

slots = num_cores

to_runs = np.ones(len(filenames), dtype=int)
cur_runs = np.zeros(num_cores, dtype=int)
done_runs = np.zeros(len(filenames), dtype=int)
# print(cur_runs)


temps = natsorted(np.array(glob.glob(commdir+"/*_plotfitter_temp_{}.txt".format(seed))))
for temp in temps:
    data = np.loadtxt(temp, dtype=str)
    try:
        if(len(data)==1):
            continue
    except TypeError:
        continue
    # print(data)

    df_temp = pd.read_csv(temp, delim_whitespace=True)

    # index = np.argwhere(df['GALAXY']==df_temp['GALAXY'].values.item()).item()
    # df

    # df = df.merge(df, df_temp, on='GALAXY', how='left')
    df.update(df[['GALAXY']].merge(df_temp, 'left'))
    # df = pd.concat([df, df_temp])

print(df.columns)
for column in df.columns:
    if(column=='GALAXY'):
        continue
    else:
        df[column] = pd.to_numeric(df[column])
    
pd.options.display.float_format = '{:.3E}'.format
print(df)

for temp in temps:
    os.remove(temp)

# print(df.dtypes)

df.to_string(savename_info, index=False)#, float_format='{:.3e}')

    

