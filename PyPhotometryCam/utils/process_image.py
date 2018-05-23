import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from PyPhotometryCam.utils.fit_func import gaussian_with_baseline

def cut_img(img,star_list_df,cut_size=10):
    name_list = []
    img_list  = []
    for i,row in star_list_df.iterrows():
        name_list.append(row['name'])
        cut_img = img[row.ypos-cut_size:row.ypos+cut_size,row.xpos-cut_size:row.xpos+cut_size]
        img_list.append(cut_img)
    return pd.DataFrame({'name':name_list,'img':img_list})

def fit_1d(img,cut_size=10,sliced=True,two_try_threshold=2.0):
    low_  = int(cut_size-3)
    high_ = int(cut_size+3)
    if(sliced):
        dd =np.sum(img[low_:high_],axis=0) 
    else:
        dd =np.sum(img,axis=0) 
    if(dd.min() <= 0):
        dd[np.where(dd <= 0)] = 1
    try:
        opt,cov = curve_fit(gaussian_with_baseline,np.arange(len(dd)),dd,(np.mean(dd),cut_size,2,np.mean(dd)),
                            sigma=np.sqrt(dd),absolute_sigma=True,bounds=([0,0,0.1,0],[np.inf,np.inf,np.inf,np.inf]))
    except:
        opt = np.array([-100,10,1,1])
        cov = np.diag(np.array([1,1,1,1]))
    chi2 = np.sum((dd - gaussian_with_baseline(np.arange(len(dd)),*opt))**2/dd)
    ndf = len(dd)  - 4
    chi2_ndf = chi2/ndf
    fit_axis =0
    if((two_try_threshold is not None) and (chi2_ndf > two_try_threshold) ):
        old_chi2_ndf, old_opt,old_cov = chi2_ndf,opt,cov 
        if(sliced):
            dd =np.sum(img[:,low_:high_],axis=1) 
        else:
            dd =np.sum(img,axis=1) 
        if(dd.min() <= 0):
            dd[np.where(dd <= 0)] = 1
        try:
            opt,cov = curve_fit(gaussian_with_baseline,np.arange(len(dd)),dd,(np.mean(dd),cut_size,2,np.mean(dd)),
                                sigma=np.sqrt(dd),absolute_sigma=True,bounds=([0,0,0.1,0],[np.inf,np.inf,np.inf,np.inf]))
        except:
            opt = np.array([-100,10,1,1])
            cov = np.diag(np.array([1,1,1,1]))
        chi2 = np.sum((dd - gaussian_with_baseline(np.arange(len(dd)),*opt))**2/dd)
        ndf = len(dd)  - 4 
        chi2_ndf = chi2/ndf
        fit_axis=1
        if(old_chi2_ndf < chi2_ndf):
            chi2_ndf = old_chi2_ndf
            opt = old_opt
            cov = old_cov
            fit_axis = 0
    return fit_axis,chi2_ndf,opt,cov
