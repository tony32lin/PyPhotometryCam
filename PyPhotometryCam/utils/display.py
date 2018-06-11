import numpy as np
import seaborn as sns
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PyPhotometryCam.utils.fit_func import gaussian_with_baseline,linfunc,ze_lin_func
from scipy.optimize import curve_fit
import logging
plt.style.use(astropy_mpl_style)

logger = logging.getLogger(__name__)

def printOneRun(output_pdf,df_with_fit,df_img,original,print_criteria=None):
    df_img_with_fit = df_img.merge(df_with_fit,on='name')

    if(print_criteria is not None):
        df_img_sel = df_img_with_fit.query(print_criteria)
    else:
        df_img_sel = df_img_with_fit
    if (len(df_img_sel) < 2):
        logger.warning('Cannot output {}. There are less than 2 stars useable.'.format(output_pdf)) 
        return

    with PdfPages(output_pdf) as pdf:

        fig = plt.figure(figsize=(8.27,11.69))
        ax = plt.subplot()        
        #show the entire picture
        show_img_with_star(original,df_img_sel,ax=ax)
        pdf.savefig()
        plt.close()
       

        fig = plt.figure(figsize=(8.27,11.69))
        gs = gridspec.GridSpec(2,1)
        gs.update( hspace=0.3,wspace =0.3)

        #show fit
    
        ax = plt.subplot(gs[0])
        ax.errorbar(df_img_sel.FLUX_B,-5/2.*np.log10(df_img_sel.Size),yerr=5/2.*df_img_sel.SizeErr/df_img_sel.Size/np.log(10),fmt='.')
        ax.set_xlabel('Magnitude (B)')
        ax.set_ylabel(r'$-\frac{5}{2}log_{10}(Signal)$')
        opt,cov = curve_fit(linfunc,df_img_sel.FLUX_B,-5/2.*np.log10(df_img_sel.Size),sigma=5/2.*df_img_sel.SizeErr/df_img_sel.Size/np.log(10),absolute_sigma=True)
        xx = np.linspace(0.8*np.min(df_img_sel.FLUX_B),
                         1.2*np.max(df_img_sel.FLUX_B),50)
        ax.plot(xx,linfunc(xx,*opt))
        # show fit against zenith angle
        ax = plt.subplot(gs[1])
        ax.errorbar(1/np.cos(np.deg2rad(90.-df_img_sel.EL)),
                    df_img_sel.FLUX_B+5/2.*np.log10(df_img_sel.Size),yerr=5/2.*df_img_sel.SizeErr/df_img_sel.Size/np.log(10),fmt='.')
        opt,cov = curve_fit(ze_lin_func,1/np.cos(np.deg2rad(90.-df_img_sel.EL)),
                            df_img_sel.FLUX_B+5/2.*np.log10(df_img_sel.Size),sigma=5/2.*df_img_sel.SizeErr/df_img_sel.Size/np.log(10),
                            absolute_sigma=True)
        xx = np.linspace(1,1.5,50)
        ax.plot(xx,ze_lin_func(xx,*opt))  
        ax.set_xlabel(r'$\frac{1}{cos(\theta_{zenith})}$')
        ax.set_ylabel(r'$M_{B}+\frac{5}{2}log_{10}(Signal)$')
        pdf.savefig()
        plt.close()

        
        #show magnitude 
        numPages = 0    
        i=0 
        for ix,row in df_img_sel.iterrows():
            if( int(i/3) >= numPages):
                numPages += 1  
                fig = plt.figure(figsize=(8.27,11.69))
                gs = gridspec.GridSpec(3, 2)
                gs.update( hspace=0.3,wspace =0.3)
             
            img = row['img']
            fit_axis = row['FitAxis']
            if(fit_axis == 0):
                dd = np.sum(img[7:13],axis=0) 
            else:
                dd = np.sum(img[:,7:13],axis=1)  
            ax = plt.subplot(gs[i % 3,0])
            img_show = ax.imshow(row['img'])
            ax.set_title(row['name'])
            ax = plt.subplot(gs[i % 3,1])
            norm = row['Size']/row['Width']/np.sqrt(2*np.pi) 
            pos  = row['FitPosX']
            width = row['Width']
            bs    = row['BaseLine']
            ax.errorbar(np.arange(len(dd)),dd,yerr=np.sqrt(dd),fmt='.')            
            ax.plot(np.linspace(0,20,100),gaussian_with_baseline(np.linspace(0,20,100),norm,pos,width,bs))

            ax.set_title('1D Gaussian Fit') 
            mg_and_size=r'$-\frac{5}{2}log_{10}(Signal) =$' + '{:.1f}'.format(-5/2.*np.log10(row.Size))
            mg_and_size = mg_and_size + '\n' + '$M_B = $' + '{:.1f}'.format(row.FLUX_B)
            if(fit_axis == 0):
                mg_and_size = mg_and_size + '\n' + 'Slice axis: X'
            else:
                mg_and_size = mg_and_size + '\n' + 'Slice axis: Y'
            ax.text(0.05, 0.91,mg_and_size,horizontalalignment='left',verticalalignment='center',
             transform = ax.transAxes,fontdict={'size':6})

            page_end = ((int((i+1)/3) >= numPages) and (numPages >0))  or (i == (len(df_img_sel) -1))
            i = i+1
            if( page_end):
                pdf.savefig()
                plt.close()

def show_img_with_star(data_cube,star_list_df,ax=None):
    if(ax is None):
        ax = plt
    data_cube_padded = data_cube.copy()
    sorted_img = np.sort(data_cube_padded.flatten())
    second_last = sorted_img[sorted_img > 0][0]
    data_cube_padded[np.where(data_cube_padded <=0)] = second_last
    img=ax.imshow(np.log10(data_cube_padded),interpolation='nearest')  
    for i,row in star_list_df.iterrows():
        ax.text(row.xpos,row.ypos,row['name'],horizontalalignment='right',verticalalignment='center',
                 fontdict={'size':10},color='red')
