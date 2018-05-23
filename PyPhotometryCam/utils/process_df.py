import numpy as np 
import pandas as pd
from os import path
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Galactic, FK4, FK5
from astropy.io import fits
from PyPhotometryCam.utils.process_image import fit_1d 

customSimbad = Simbad()
customSimbad.add_votable_fields('ra(deg)','dec(deg)','flux(B)','flux(V)')
customSimbad.remove_votable_fields('coordinates')


def parse_fname(filename):
    path_base_name = path.splitext(path.basename(filename))[0]
    tel_str,filter_str,date_str,time_str = path_base_name.split('_')
    tel_id = int(tel_str[1])
    filter_id = int(filter_str[1])
    day = int(date_str[0:2])
    month = int(date_str[2:4])
    year = 2000 + int(date_str[4:6])
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    parsed_time = '{year:d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}'.format(year=year,
                                                                                               month=month,
                                                                                               day=day,
                                                                                               hour=hour,
                                                                                               minute=minute,
                                                                                               second=second)
    return tel_id,filter_id,parsed_time

def add_elevation(df,time_str,location):
    utcoffset = -7*u.hour  
    time = Time(time_str) -utcoffset
    elevation_arr = []
    azimuth_arr = []
    for i,row in df.iterrows():
        coord = SkyCoord(row['RA']*u.deg,row['DEC']*u.deg,frame='icrs')
        out=coord.transform_to(AltAz(obstime=time,location=location))
        elevation_arr.append(out.alt.to_value(unit=u.deg))
        azimuth_arr.append(out.az.to_value(unit=u.deg))
    df['EL'] = elevation_arr
    df['AZ'] = azimuth_arr
    return df

def get_star_list(fbase_name,filter_stars=True,starlist_dir='StarCoordinates/'):
    radec = starlist_dir + fbase_name +'.rdls'
    pixellist = starlist_dir+  fbase_name +'-indx.xyls'
    hdul = fits.open(radec)[1].data
    hdul2 = fits.open(pixellist)[1].data
    star_list = []
    for ii in range(0,len(hdul['RA'])):
        sky_coordinates = SkyCoord('{}  {} '.format(hdul['RA'][ii],hdul['DEC'][ii]), unit = (u.deg , u.deg ) )
        result = customSimbad.query_region(sky_coordinates) 
        #print(result[0][0])

        # Select Stars with Visual Magnitude < 8
        if result[0][4] < 8:
            star_list.append([result[0][0],int(hdul2["X"][ii]),int(hdul2["Y"][ii]),result[0][4],result[0][3],result[0][1],result[0][2]])
    if(filter_stars):
        star_list = np.array(star_list)
        #Reduce FOV to exclude stars behind telescope.
        star_list = star_list[(star_list[:,1].astype(int) >= 400) & (star_list[:,2].astype(int) <= 400)]
        final = np.copy(star_list)
        flag = []
        for jj in range(len(star_list)):
            to_delete = np.where((star_list[:,1].astype(int) >= star_list[jj,1].astype(int) -15)
                      & (star_list[:,1].astype(int) <= star_list[jj,1].astype(int) +15)
                      & (star_list[:,2].astype(int) >= star_list[jj,2].astype(int) -15)
                      & (star_list[:,2].astype(int) <= star_list[jj,2].astype(int) +15))
            if len(to_delete[0]) > 1:
                flag.append(to_delete)
        flat_list = [item for sublist in flag for subsublist in sublist for item in subsublist]
    
        final_star_list = np.delete(final,list(set(flat_list)),axis = 0)
    else:
        final_star_list = np.array(star_list)
    df = pd.DataFrame({'name':final_star_list[:,0].astype(str),'xpos':final_star_list[:,1].astype('int'),
                       'ypos':final_star_list[:,2].astype('int'),
                       'FLUX_V':final_star_list[:,3].astype(float),
                       'FLUX_B':final_star_list[:,4].astype(float),
                       'RA':final_star_list[:,5].astype(float),
                       'DEC':final_star_list[:,6].astype(float)})
    return df

def get_img_and_star_list(fname,filter_stars=True,starlist_dir='StarCoordinates/'):
    bname = path.splitext(path.basename(fname))[0]
    data_cube = fits.getdata(fname,0)
    header    = fits.getheader(fname)
    df = get_star_list(bname,filter_stars=filter_stars,starlist_dir=starlist_dir)
    df['Exposure'] = np.ones(len(df),dtype=float)*header['EXPTIME']
    df['CCDTEMP'] = np.ones(len(df),dtype=float)*header['CCD-TEMP']
    return data_cube,df

def get_fitted_size(df_img,two_try_threshold=2.0,cut_size=10):
    size_arr =[]
    size_err_arr = []
    width_arr = []
    width_err_arr = []
    pos_arr =[]
    pos_err_arr =[]
    chi2_arr = []
    name_arr =[]
    bs_arr =[]
    bs_err_arr = []
    fit_axis_arr = []
    for i,row in df_img.iterrows():
        fit_axis,chi2ndf,opt,cov = fit_1d(row['img'],two_try_threshold=two_try_threshold,cut_size=cut_size)
        chi2_arr.append(chi2ndf)
        name_arr.append(row['name'])
        size_arr.append(opt[0]*np.sqrt(2*np.pi)*opt[2])
        norm_err_sq = cov[0,0]
        sigma_err_sq = cov[2,2]
        norm_err_sq = 2*np.pi*(norm_err_sq*opt[2]**2+opt[0]**2*sigma_err_sq)
        size_err_arr.append(np.sqrt(norm_err_sq))
        width_arr.append(opt[2])
        width_err_arr.append(np.sqrt(cov[2,2]))
        pos_arr.append(opt[1])
        pos_err_arr.append(np.sqrt(cov[1,1]))
        bs_arr.append(opt[3])
        bs_err_arr.append(np.sqrt(cov[3,3]))
        fit_axis_arr.append(fit_axis)
    return pd.DataFrame({'name':name_arr,'Size':size_arr,'SizeErr':size_err_arr,'Chi2':chi2_arr,
                         'BaseLine':bs_arr,'BaseLineErr':bs_err_arr,'Width':width_arr,'WidthErr':width_err_arr,
                         'FitPosX':pos_arr,'FitPosXErr':pos_err_arr,'FitAxis':fit_axis_arr})
