import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Galactic, FK4, FK5
import numpy as np 
from astropy.io import fits
from scipy.optimize import curve_fit
import os.path as path
import os 

from PyPhotometryCam.utils.process_df import *
from PyPhotometryCam.utils.process_image import *
import PyPhotometryCam.utils.display as display 
import matplotlib.pyplot as plt
import logging 

logger = logging.getLogger(__name__)

class timeSeriesAnalyzer:
    def __init__(self,star_list_dir,flat_field_dict):
        self.__star_list_dir__ = star_list_dir
        self.__flat_field_dict__ = flat_field_dict
        self.__all_star_analysis_result__ = None
        self.__ts_analysis_result__ = None
        self.__analyzer__ = imageAnalyzer(star_list_dir,flat_field_dict) 

    def __check__(self,file_list):
        tel_filter_list = []
        for tel_id,filter_id,time_str in map(parse_fname,file_list):
            tel_filter_list.append((tel_id,filter_id))
        tel_filter_list = set(tel_filter_list)
        for tel,ff in tel_filter_list:
            if(tel in self.__flat_field_dict__.keys()):
                if(ff in self.__flat_field_dict__[tel].keys()):
                    continue
            raise KeyError('Flatfiled image for T{:d} F{:d}'.format(tel,ff))

    def analyze(self,file_list,run_pdf_dir=None,
                     criteria='(Chi2 < 2 ) & (Size >0 ) & (Size/SizeErr > 2.0 ) & (Width < 2.5)'):
        self.__check__(file_list)
        count = 0
        df_arr = []
        for f in file_list:
            if(not path.exists(f)):
                logger.warning(f + ' does not exist. File skipped !')
                continue
            # check astrometry result exist
            fbase_name = path.splitext(path.basename(f))[0]
            radec = self.__star_list_dir__ + fbase_name +'.rdls'
            pixellist = self.__star_list_dir__+  fbase_name +'-indx.xyls'
            if((not path.exists(radec)) or (not path.exists(pixellist))):
                logger.warning(f + ' does not have astrometry analysis result !')
                continue
               
            if(run_pdf_dir is not None):
                os.makedirs(run_pdf_dir,exist_ok=True)                  
                outpdf_name = run_pdf_dir + '/' + path.splitext(path.basename(f))[0] + '.pdf' 
            df_result = self.__analyzer__.analyze(f,output_pdf=outpdf_name,print_criteria=criteria)                         
            df_result['PID'] = np.ones(len(df_result),dtype=int)*count
            df_arr.append(df_result)
            logger.info(f + ' processed.')
            count += 1 

        if len(df_arr) ==0 :
            raise Exception('No file analized.') 
        else:
            df_orig = pd.concat(df_arr)             
            df_sel = df_orig.query(criteria)
        throughput_arr = []
        throughput_err_arr = []
        one_over_cos_arr = []
        one_over_cos_err_arr = []
        exposure_arr = []
        time_arr =[]
        for i in df_sel.PID.unique():
            dd = df_sel[df_sel.PID == i]
            ss = dd.Size/dd.Exposure
            ss_err = dd.SizeErr/dd.Exposure
            cos_ze = np.cos(np.deg2rad(90.-dd.EL))
            throughput     = dd.FLUX_B + 5./2.*np.log10(ss) 
            throughput_err = 5./2.*ss_err/ss/np.log(10)
            weight = 1/(throughput_err**2)
            mean_throughput = np.sum(weight*throughput)/np.sum(weight)
            mean_throughput_err = np.sqrt(1/(np.sum(weight)))            
            exposure_arr.append(dd.Exposure.iloc[0])
            throughput_arr.append(mean_throughput)
            throughput_err_arr.append(mean_throughput_err)
            time_arr.append(dd.Time.iloc[0])
            one_over_cos_arr.append(np.mean(1/cos_ze))
            one_over_cos_err_arr.append(np.std(1/cos_ze)/np.sqrt(len(dd)))
        self.__all_star_analysis_result__ = df_sel
        self.__ts_analysis_result__       = pd.DataFrame({'Time':time_arr,'Exposure':exposure_arr,'throughput':throughput_arr,
                                                          'throughputErr':throughput_err_arr,'oneOverCos':one_over_cos_arr,
                                                          'oneOverCosErr':one_over_cos_err_arr})             

    def get_results(self):
        if(self.__all_star_analysis_result__ is not None):
            return self.__ts_analysis_result__.copy(),self.__all_star_analysis_result__.copy()
        else:
            raise Exception('No analysis done yet!')

    def show_result(self):
        if(self.__all_star_analysis_result__ is None):
            raise Exception('No analysis done yet!')
        ts_result = self.__ts_analysis_result__
        all_star  = self.__all_star_analysis_result__
        ss = all_star.Size/all_star.Exposure
        ss_err = all_star.SizeErr/all_star.Exposure

        plt.figure()
        plt.errorbar(all_star.FLUX_B,-5./2.*np.log10(ss),yerr=5./2.*ss_err/ss/np.log(10),fmt='.')
        ax = plt.gca()
        ax.set_ylabel(r'$-\frac{5}{2} log_{10} (\frac{Signal}{T_{exposure}})$')
        ax.set_xlabel(r'$M_{B}$')
        ax.set_title('Signal VS Magnitude of All Stars')

        plt.figure()
        xerr = ts_result.Exposure.map(lambda x: pd.Timedelta(seconds=x))
        plt.errorbar(ts_result.Time.values,ts_result.throughput,yerr=ts_result.throughputErr,fmt='.')
        ax = plt.gca()
        ax.set_ylabel(r'$M_{B}+\frac{5}{2} log_{10} (\frac{Signal}{T_{exposure}})$')
        ax.set_xlabel(r'$Time (GMT -7H)$')
        ax.set_title('Throughput Time Series')
                    
        plt.figure()
        plt.errorbar(all_star.Time.values,all_star.BaseLine/all_star.Exposure,yerr=all_star.BaseLineErr/all_star.Exposure,fmt='.')             
        ax = plt.gca()
        ax.set_ylabel(r'Background (DC/$T_{exposure}$)')
        ax.set_xlabel(r'$Time (GMT -7H)$')
        ax.set_title('Background Level Time Series')
       
class imageAnalyzer:
    def __init__(self,star_list_dir,flat_field_dict):
        self.__star_list_dir__ = star_list_dir
        self.__flat_field_dict__ = flat_field_dict
        self.__location__ = EarthLocation.of_site('flwo')   

    def analyze(self,fname,output_pdf=None,print_criteria=None):
        time = []
        data_cube,df = get_img_and_star_list(fname,filter_stars=True,starlist_dir=self.__star_list_dir__)
        tel,f,time_str = parse_fname(fname)      
        df = add_elevation(df,time_str,self.__location__)
        for i in range(len(df)):
            time.append(pd.to_datetime(time_str))

        ff_data = self.__flat_field_dict__[tel][f]
        corrected = data_cube/ff_data
        
        df_img = cut_img(corrected,df,cut_size=10)
        df_fit = get_fitted_size(df_img,two_try_threshold=2.0,cut_size=10)
        df_with_fit =df.merge(df_fit,on='name')
        df_with_fit['Time'] = time
        if(output_pdf is not None): 
            display.printOneRun(output_pdf,df_with_fit,df_img,original=corrected,
                                print_criteria=print_criteria)
        del df_img        
        return df_with_fit 

     
