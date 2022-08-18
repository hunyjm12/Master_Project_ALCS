import pip
from astropy.io import fits
import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
from functools import reduce
#import mplcursors
import math
import pandas as pd
import astropy.units as u
import sys
from astropy.coordinates import SkyCoord
import csv
import multiprocessing as mp
from astropy.nddata import Cutout2D
from astropy import units as u
#from adjustText import adjust_text
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib as mpl
from functools import reduce
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import os
import glob
import time
from astropy.table import Table
from numpy import inf
#%%
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 25
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

#%%
def clean_data(array):
    array[array<0]=0
    array[array== inf]=0
    array[array== -inf]=0
    array[np.isnan(array)]=0
    return array

def uvj_constants(z):
    if (z>2.0 and z<3.5):
        redshift='2.0<z<3.5'
        x1 = 1.4
        y1 = 1.3
        const = 0.59
    elif (z>1.5 and z<2.0):
        redshift='1.5<z<2.0'
        x1 = 1.5
        y1 = 1.3
        const = 0.59
    elif (z>0 and z<1.5):
        x1 = 1.6
        y1 = 1.3
        if z<0.5:
            redshift='0<z<0.5'
            const = 0.69
        elif z>0.5:
            redshift='0.5<z<1.5'
            const = 0.59
    x_right = (y1 - const) / 0.88  # right point of horizontal line
    x_horizontal = np.linspace(0, x_right, 50)  # horizontal line
    y_horizontal = np.ones(x_horizontal.size)*y1     # horizontal line
    x_diagonal = np.linspace(x_right, x1, 50)   # diagonal line
    y_diagonal = 0.88*x_diagonal + const       # diagonal line
    y_lower = y_diagonal[-1]    # lower point of vertical line
    y_vertical = np.linspace(y_lower, 4, 50)  # vertical line
    x_vertical = np.ones(y_vertical.size)*x1      # vertical line
    return redshift, x_diagonal, y_diagonal, x_horizontal, y_horizontal, x_vertical, y_vertical, const

# Schreiber 2015
def trace(m, z, a0=1.5, a1=0.3, a2=2.5, m0=0.5, m1=0.36):
    r = np.log10(1 + z)
    M = np.log10(m/1e9)
    p = M-m1-a2*r
    p[p<0] = 0
    return M - m0 + a0*r - a1*p**2

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions,text, axis,txt_width,txt_height):
    for x,y,t, txt in zip(x_data, y_data, text_positions, text):
        axis.text(x - txt_width, 1.01*t, '%d'%int(txt),rotation=0, color='hotpink')
        if y != t:
            axis.arrow(x, t,0,y-t, color='blue',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width, head_length=txt_height*0.5, 
                       zorder=10,length_includes_head=True)

#%%
#========================================== ALMA detected objects ===========================================

def matching(id_hst, id_alma, flux_alma, flux_err_alma, snr_alma, redshift, catalog1_ra, catalog1_dec, catalog2_ra, catalog2_dec, intended_sep=3): 
    # find HST counterparts of ALMA objects
    # catalog1 = HST, catlog2 = ALMA, in this case
    coor_hst=[]
    ra_detected=[]
    dec_detected=[]
    id_alma_detected=[]
    delta_ra=[]
    flux_alma_detected=[]
    err_alma_detected=[]
    index_hst = []
    snr_detected=[]
    id_detected=[]
    delta_dec=[]
    id_hst_detected=[]
    z=[]
    separation = []
    for i in range(id_alma.size):
        if snr_alma[i]>4:
            c1 = SkyCoord(catalog1_ra*u.deg, catalog1_dec*u.deg, frame='fk5')
            c2 = SkyCoord(catalog2_ra[i]*u.deg, catalog2_dec[i]*u.deg, frame='fk5')
            sep = c1.separation(c2)
            sep_arcs = sep.arcsecond
            detected = catalog1_ra[np.where(sep_arcs < intended_sep)] # find corresponding objects in HST catalogue
            sep1 = sep_arcs[np.where(sep_arcs < intended_sep)]
            if sep1.size>0:
                if sep.size>1:        # if more than one objects found in catalogue, 
                    a = np.argmin(sep1)     # select out the coordinate corresponding to the smallest separation.
                    detected = detected[a]
                    sep_min = sep1[a]
                index=np.where(catalog1_ra==detected)  # index of this object, in HST catalog
                'if int(*c) in final:   # check if the object fufill those cuts above'
                id_alma_detected=np.append(id_alma_detected, id_alma[i])
                coor_hst=np.append(coor_hst, int(*index))
                index_hst = np.append(index_hst, *index[0])   # zero-base index
                ra_detected=np.append(ra_detected, catalog2_ra[i])
                z=np.append(z, redshift[index])
                flux_alma_detected = np.append(flux_alma_detected, flux_alma[i])
                err_alma_detected = np.append(err_alma_detected, flux_err_alma[i])
                snr_detected = np.append(snr_detected, snr_alma[i])
                id_hst_detected = np.append(id_hst_detected, id_hst[index])   # id, some are missing
                dec_detected=np.append(dec_detected, catalog2_dec[i])
                id_detected=np.append(id_detected, id_alma[i])
                delta_ra = np.append(delta_ra, catalog1_ra[index] - catalog2_ra[i])
                delta_dec = np.append(delta_dec, catalog1_dec[index] - catalog2_dec[i])
                separation = np.append(separation, sep_min)
            else:
                continue
        else:
            continue
    coor_hst = coor_hst.astype(int)
    return id_alma_detected, ra_detected, dec_detected, separation, coor_hst, catalog1_ra[coor_hst], catalog1_dec[coor_hst], z, delta_ra, delta_dec, id_hst_detected, index_hst, flux_alma_detected, err_alma_detected, snr_detected

#%%

path2 = '/home/yjm/miniconda/Outputs'
os.chdir(path2)
files_list = glob.glob('*')

def main_body(file_name):
    try:
        os.chdir(path2+'/'+file_name)
    except:
        print(file_name)
        return
    '''
    if file_name!='a0102':
        if file_name!='a370':
            return 1
    '''
    lists = glob.glob('*master.fits')
    if len(lists)==0:
        lists = glob.glob('*full_table.fits')
    try:
        name=lists[0]
    except:
        print(file_name, 'no file')
        return 1
    
    hdulist2 = fits.open(name)
    data3=hdulist2[1].data
    df = pd.read_csv('cluster_redshift.txt', header=None)
    cluster_redshift = df.to_numpy()[:,0].astype(float)
    # redshift: https://github.com/dawn-cph/alcs-clusters/blob/master/v1.0/fields.md
    A = pd.read_csv('../../ALCS_catalog_comb.dat',comment='#',sep='\s+',header=None, index_col=0, usecols=[0, 1, 2, 3, 10, 19, 24], names=['cluster', 'id', 'ra','dec', 'snr', 'flux', 'flux_err'])
    regex_str = '('+file_name[1:]+')'
    match = (A.filter(regex=regex_str, axis=0))
    
    id_alma = match.to_numpy()[:,0].astype(int)
    ra_alma = match.to_numpy()[:,1].astype(float)
    dec_alma = match.to_numpy()[:,2].astype(float)
    snr_alma = match.to_numpy()[:,3].astype(float)
    flux_alma = match.to_numpy()[:,4].astype(float)
    flux_err_alma = match.to_numpy()[:,5].astype(float)
    try:
        hband_flux = data3['f160w_flux']
        hband_err = data3['f160w_err']
        hband_snr = hband_flux / hband_err
        coor_hband = np.where(hband_snr>3)
        H = -2.5 * np.log10(hband_flux) + 23.9
    except:
        hband_flux = data3['f160w_tot_1']
        hband_err = data3['f160w_etot_1']
        hband_snr = hband_flux / hband_err
        coor_hband = np.where(hband_snr>3)
        H = -2.5 * np.log10(hband_flux) + 23.9
    try:
        b_flux = data3['restB']
        b_err = data3['restB_err']
        u_flux = data3['restU']
        u_err = data3['restU_err']
        v_flux = data3['restV']
        v_err = data3['restV_err']
        id_hst = data3['id']
        j_flux = data3['restJ']
        magnification = data3['mu']
        j_err = data3['restJ_err']
        z = data3['z_phot']
        Av = data3['Av']
        num = data3['nusefilt']
        try:
            sfr = data3['SFR']
        except:
            sfr = data3['sfr']
        nuv_flux = data3['restNUV'] # uJy
        nuv_err = data3['restNUV_err']
        irac_ch1_flux = data3['irac_ch1_flux']
        irac_ch2_flux = data3['irac_ch2_flux']
        sfr_16 = data3['sfr_p'][:,1]
        sfr_84 = data3['sfr_p'][:,3]
        mass_16 = data3['mass_p'][:,1]
        mass_84 = data3['mass_p'][:,3]
        Av_16 = data3['Av_p'][:,1]
        Av_84 = data3['Av_p'][:,3]
        irac_ch1_err = data3['irac_ch1_err']
        irac_ch2_err = data3['irac_ch2_err']
        mass = data3['mass']
        
        irac_ch1_snr = irac_ch1_flux / irac_ch1_err
        irac_ch2_snr = irac_ch2_flux / irac_ch2_err
        z_16 = data3['z160']
        z_84 = data3['z840']
        ra = data3['ra']
        dec = data3['dec']
    except Exception as e:
        print(file_name+' fuck '+str(e))
        
        return

    
    
    try:
        id_detected, ra_detected, dec_detected, d2d, idxx, c1_ra, c1_dec, z_alma, delta_ra, delta_dec, id_hst_detected, index_in_hst, flux_alma_detected, err_alma_detected, snr_detected = matching(id_hst, id_alma, flux_alma, flux_err_alma, snr_alma, z, ra, dec, ra_alma, dec_alma)
    except Exception as e:
        print(file_name)
        print(e)
    coor_behind = np.where(z_16[idxx] > (cluster_redshift + 0.1))
    coor_before = np.where(z_84[idxx] < (cluster_redshift - 0.1))
    coor_within = np.where((z_16[idxx] < (cluster_redshift + 0.1)) | (z_84[idxx] > (cluster_redshift - 0.1)))
    flag_position = np.ones(z_16[idxx].size)
    # 0=before, 1=inside, 2=behind
    flag_position[coor_before] = 0
    flag_position[coor_behind] = 2
    lamb_rest = 1.2/(z_alma + 1)*1e3
    
    U = -2.5 * np.log10(u_flux) + 23.9 # * 1e-29 / 1.79e-20)  # / 0.77e-14 * 1e26)  erg/cm^2 --> J/m^2 --> uJy
    U_err = 0.434 * u_err / u_flux * 2.5
    B = -2.5 * np.log10(b_flux) + 23.9 # * 1e-29 / 4.063e-20) # / -0.12e-14 * 1e26)
    B_err = 0.434 * b_err / b_flux * 2.5
    V = -2.5 * np.log10(v_flux) + 23.9 #* 1e-29 / 2.636e-20)
    V_err = 0.434 * v_err / v_flux * 2.5
    J = -2.5 * np.log10(j_flux) + 23.9 # * 1e-29 / 1.589e-20)# / 0.899e-14 * 1e26)
    J_err = 0.434 * j_err / j_flux * 2.5

    UB = U - B
    ub_err = np.sqrt(U_err**2 + B_err**2)
    BV = B - V
    bv_err = np.sqrt(V_err**2 + B_err**2)
    VJ = V - J
    vj_err = np.sqrt(V_err**2 + J_err**2)
    UV = U - V
    uv_err = np.sqrt(U_err**2 + V_err**2)
    
    t = Table([id_detected, index_in_hst, id_hst_detected, ra_detected,
               dec_detected, z_alma, Av[idxx], sfr[idxx], nuv_flux[idxx], nuv_err[idxx],
               mass[idxx], z_16[idxx], z_84[idxx], flag_position,
               lamb_rest, flux_alma_detected, err_alma_detected, snr_detected, d2d, magnification[idxx], uv_err[idxx],
               vj_err[idxx], sfr_16[idxx], sfr_84[idxx], mass_16[idxx], mass_84[idxx],
               Av_16[idxx], Av_84[idxx], hband_snr[idxx]],
               names=('id_alma', 'id_hst', 'id_eazy', 'ra', 'dec', 'z', 'Av', 'sfr','NUV','NUV_err',
                      'mass', 'z_16', 'z_84', 'flag_position', 'lamb_alma',
                      'flux_1.2mm', 'flux_1.2mm_err','snr_1.2mm', 'separation', 'magnification',
                      'uv_err', 'vj_err', 'sfr_eazy_16', 'sfr_eazy_84', 'mass_16', 'mass_84',
                      'Av_16', 'Av_84', 'f160w_snr')
             )
    
    t.write(name.split('_')[0]+'_alma_detected.fits', format='fits', overwrite=True)
    
    mass = mass / magnification
    mass_16 = mass_16 / magnification
    mass_84 = mass_84 / magnification
    sfr = sfr / magnification
    sfr_16 = sfr_16 / magnification
    sfr_84 = sfr_84 / magnification
    # UVJ

    uv_snr = UV / uv_err
    vj_snr = VJ / vj_err

    z_bins = np.array([[0, 0.5, 0.25],[0.5, 1.5, 1.0],[1.5, 2.0, 1.75], [2.0, 2.5, 2.25], [2.5, 3.5, 3]])
    color='tomato'
    fig,axes=plt.subplot_mosaic([['1','2', '3','4', 'a'], ['5', '6', '7', '8', 'b'], ['9','10','11','12', 'c']], empty_sentinel="X", figsize=(35, 21))
    axes = list(axes.values())
    #axes.tight_layout()
    num_bins=30
    index=5
    for i in range(index):
        coor_z = np.where((z>z_bins[i,0]) & (z<z_bins[i,1]))
        coor_z_alma = np.where((z_alma>z_bins[i,0]) & (z_alma<z_bins[i,1]))
        coor_z_idxx = reduce(np.intersect1d,(coor_z, idxx))
        coor_before2 = np.where(z_84[coor_z_idxx] < (cluster_redshift - 0.1))
        #coor_behind2 = reduce(np.intersect1d,(coor_z, coor_behind))
        coor_behind2 = np.where(z_16[coor_z_idxx] > (cluster_redshift + 0.1))
        coor_within2 = np.where((z_16[coor_z_idxx] < (cluster_redshift + 0.1)) & (z_84[coor_z_idxx] > (cluster_redshift - 0.1)))
        ybins = np.linspace(0, 5, num=num_bins)
        xbins = np.logspace(3, 16, num=num_bins)
        counts, _, _ = np.histogram2d(mass[coor_z], Av[coor_z], bins=(xbins, ybins))
        sc=axes[i].pcolormesh(xbins, ybins, counts.T, cmap='binary')
        cbar=plt.colorbar(sc, ax=axes[i])
        cbar.set_label('n')
        axes[i].set_ylim(0,5)
        #axes[i].scatter(mass[coor_z_idxx], Av[coor_z_idxx], s=60, marker='x', c='red')
        axes[i].scatter(mass[coor_z_idxx][coor_before2], Av[coor_z_idxx][coor_before2], s=10, label='foreground', c='blue', zorder=15)
        axes[i].scatter(mass[coor_z_idxx][coor_behind2], Av[coor_z_idxx][coor_behind2], s=10, label='background', c='red', zorder=15)
        axes[i].scatter(mass[coor_z_idxx][coor_within2], Av[coor_z_idxx][coor_within2], s=10, label='within cluster', c='orange', zorder=15)
        mass_err = np.vstack([mass[coor_z_idxx] - mass_16[coor_z_idxx], mass_84[coor_z_idxx] - mass[coor_z_idxx]])
        Av_err = np.vstack(([Av[coor_z_idxx] - Av_16[coor_z_idxx], Av_84[coor_z_idxx] - Av[coor_z_idxx]]))
        mass_err = clean_data(mass_err)
        Av_err = clean_data(Av_err)
        axes[i].errorbar(mass[coor_z_idxx], Av[coor_z_idxx], xerr=mass_err, yerr=Av_err, c='blue', capsize=7, ls='none')
        #axes[i].set_ylim(0,5)
        axes[i].set_xscale('log')
        axes[i].set_title(str(z_bins[i,0])+'<z<'+str(z_bins[i,1])+', <z>='+str(z_bins[i,2]))
        axes[i].grid()
        axes[i].minorticks_on()
        axes[i].legend(prop={'size': 8})
        axes[i].set_xlabel(r'$ M_*$ [$M_{\odot}$]')
        axes[i].set_ylabel('Av [mag]')
        text=[]
        
        for j, txt in enumerate(id_detected[coor_z_alma]):
            if math.isnan(mass[coor_z_idxx][j]):
                print('mass NaN')
            if math.isnan(np.log10(sfr[coor_z_idxx][j])):
                print('sfr NaN')
                continue
            txt = axes[i].text(x=mass[coor_z_idxx][j], y=Av[coor_z_idxx][j],s=int(txt), c='hotpink')
            text.append(txt)
        #adjust_text(text, arrowprops=dict(arrowstyle='->', color='red'))
        
 
        mass2 = np.logspace(4, 13, 100)
        mass_sorted = mass2
        log10_sfr_ms0 = trace(mass_sorted, z_bins[i,2])
        log10_sfr_ms0 = np.sort(log10_sfr_ms0)
        #log10_sfr_ms1 = np.sort(log10_sfr_ms1)
        
        axes[i+index].plot(mass_sorted/1.698, 10**log10_sfr_ms0, label=('main sequence z='+str(z_bins[i,2])), c='royalblue', zorder=11)
        #axes[1].scatter(np.log10(mass2/1e9)-0.23, log10_sfr_ms0, label='main sequence z=0', c='tomato')
        axes[i+index].fill_between(mass_sorted/1.698, 10**(log10_sfr_ms0-0.3), 10**(log10_sfr_ms0+0.3), facecolor='gainsboro', zorder=10) # 10^0.23 = 1.698
        ybins = np.logspace(-7, 7, num=num_bins)
        xbins = np.logspace(3, 16, num=num_bins)
        counts, _, _ = np.histogram2d(mass[coor_z], sfr[coor_z], bins=(xbins, ybins))
        sc=axes[i+index].pcolormesh(xbins, ybins, counts.T, cmap='binary')
        cbar=plt.colorbar(sc, ax=axes[i+index])
        cbar.set_label('n')
        #axes[i+index].scatter(mass[coor_z_idxx], sfr[coor_z_idxx], s=60, marker='x', c='red', zorder=15)
        axes[i+index].scatter(mass[coor_z_idxx][coor_before2], sfr[coor_z_idxx][coor_before2], s=10, label='foreground', zorder=15, c='blue')
        axes[i+index].scatter(mass[coor_z_idxx][coor_behind2], sfr[coor_z_idxx][coor_behind2], s=10, label='background', zorder=15, c='red')
        axes[i+index].scatter(mass[coor_z_idxx][coor_within2], sfr[coor_z_idxx][coor_within2], s=10, label='within cluster', zorder=15, c='orange')
        sfr_err = np.vstack(([sfr[coor_z_idxx] - sfr_16[coor_z_idxx], sfr_84[coor_z_idxx] - sfr[coor_z_idxx]]))
        sfr_err = clean_data(sfr_err)
        axes[i+index].errorbar(mass[coor_z_idxx], sfr[coor_z_idxx], xerr=mass_err, yerr=sfr_err, c='blue', capsize=7, zorder=13, ls='none')
        axes[i+index].set_xlabel(r'$ M_*$ [$M_{\odot}$] ')
        axes[i+index].minorticks_on()
        text1=[]
        for j, txt in enumerate(id_detected[coor_z_alma]):
            if math.isnan(mass[coor_z_idxx][j]):
                print('mass NaN')
            if math.isnan(np.log10(sfr[coor_z_idxx][j])):
                print('sfr NaN')
                continue
            txt = axes[i+index].text(x=mass[coor_z_idxx][j], y=sfr[coor_z_idxx][j],s=int(txt), c='hotpink', zorder=15)
        
        '''for i, txt in enumerate(id_detected):
            if math.isnan(mass[idxx][i]):
                print('mass NaN')
            if math.isnan(np.log10(sfr[idxx][i])):
                print('sfr NaN')
                continue
            txt1=axes[1].text(x=np.log10(mass[idxx][i]/1e9), y=np.log10(sfr[idxx][i]), s=int(txt), c='hotpink')
            text1.append(txt1)'''
        axes[i+index].set_yscale('log')
        axes[i+index].set_xscale('log')
        axes[i+index].set_ylabel(r'$SFR [M_{\odot} / yr]$', labelpad=10)
        axes[i+index].legend(prop={'size': 8})
        axes[i+index].grid()
        
        ybins = np.linspace(-1, 3, num=30)
        xbins = np.linspace(-1, 3, num=30)
        counts, _, _ = np.histogram2d(VJ[coor_z], UV[coor_z], bins=(xbins, ybins))
        sc=axes[i+index*2].pcolormesh(xbins, ybins, counts.T, cmap='binary')
        cbar=plt.colorbar(sc, ax=axes[i+index*2])
        cbar.set_label('n')
        redshift, x, y, x_2, y_2, x_4, y_4, const = uvj_constants( z_bins[i,2] )
        
        coor_vj = np.where(VJ[coor_z]<x_4[0])
        coor_uv = np.where(UV[coor_z]>y_2[0])
        slope = (UV[coor_z]-const)/VJ[coor_z]
        coor_diagonal = np.where(slope>0.88)
        coor_quiescent = reduce(np.intersect1d, (coor_vj, coor_uv, coor_diagonal))
        num_quiescent = coor_quiescent.size
        
        axes[i+index*2].set_title('number of quiescent galaxies: '+str(num_quiescent))
        axes[i+index*2].plot(x, y, color=color, label=redshift, zorder=5)
        axes[i+index*2].plot(x_2 ,y_2, color=color, zorder=5)
        axes[i+index*2].plot(x_4, y_4, color=color, zorder=5)
        #plt.axes.xaxis.set_visible(False)
        #plt.axes.yaxis.set_visible(False)
        #axes[i+index*2].scatter(VJ[coor_z_idxx], UV[coor_z_idxx], marker='x', s=60, c='red')
        axes[i+index*2].scatter(VJ[coor_z_idxx][coor_before2], UV[coor_z_idxx][coor_before2], s=10, label='foreground', c='blue')
        axes[i+index*2].scatter(VJ[coor_z_idxx][coor_behind2], UV[coor_z_idxx][coor_behind2], s=10, label='background', c='red')
        axes[i+index*2].scatter(VJ[coor_z_idxx][coor_within2], UV[coor_z_idxx][coor_within2], s=10, label='within cluster', c='orange')
        axes[i+index*2].errorbar(VJ[coor_z_idxx], UV[coor_z_idxx], xerr=vj_err[coor_z_idxx], yerr=uv_err[coor_z_idxx], c='blue', capsize=7, zorder=1, ls='none')
        '''
        for i in range(id_detected.size):
            ax.text(VJ[idxx][i], UV[idxx][i]+0.2, id_detected[i])
        '''
        text2=[]
        for j, txt in enumerate(id_detected[coor_z_alma]):
            if math.isnan(VJ[idxx][j]):
                print('VJ NaN')
                continue
            if math.isnan(UV[idxx][j]):
                print('UV NaN')
                continue
            txt2 = axes[i+index*2].text(x=VJ[coor_z_idxx][j], y=UV[coor_z_idxx][j], s=int(txt), c='hotpink', zorder=15)

        #adjust_text(text2, arrowprops=dict(arrowstyle='->', color='red'))
        axes[i+index*2].grid()
        axes[i+index*2].minorticks_on()
        axes[i+index*2].set_xlim(0,4)
        axes[i+index*2].set_ylim(0,4)
        axes[i+index*2].set_xlabel('V - J')
        axes[i+index*2].set_ylabel('U - V')
        axes[i+index*2].legend(prop={'size': 8})
    
    path3='/home/yjm/miniconda/PICS/'+file_name
    os.chdir(path3)
    plt.savefig(file_name+'_UVJ.pdf')
    plt.close()
    return mass,mass_16, mass_84, z, Av, Av_16, Av_84, sfr, sfr_16, sfr_84, z_alma, UV, VJ, uv_err, vj_err, id_detected, idxx, z_16,z_84, cluster_redshift


