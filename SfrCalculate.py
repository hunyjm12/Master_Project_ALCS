import glob
import pandas as pd
import scipy.constants as scc
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import scipy
import scipy.stats as stats
from scipy.integrate import quad
from matplotlib.offsetbox import AnchoredText
from astropy.io import fits
from astropy.table import Table
import warnings
import os

def MS(m, z, a0=1.5, a1=0.3, a2=2.5, m0=0.5, m1=0.36):
    r = np.log10(1 + z)
    M = np.log10(m/1e9)
    p = M-m1-a2*r
    p[p<0] = 0
    return M - m0 + a0*r - a1*p**2

def nuv_to_sfr(nuv_flux, redshift):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    d_L = cosmo.luminosity_distance(redshift).to(u.cm)
    L_v = 4*3.14*d_L.value**2*nuv_flux   # nuv_flux is in rest frame
    # L_UV = 1.5 * nu(2800) *Lv(2800) 
    L_UV = 1.5 * 3e8 / 2.8e-7 * L_v / 3.82e33
    SFR_UV = 2.2 * 1.09e-10 * L_UV
    return SFR_UV, L_UV

def main_body(file_name):
    path = '/home/yjm/miniconda/Outputs'    
    if file_name!='overall_UVJ.png':
        os.chdir(path+'/'+file_name)
    df = pd.read_csv('../../cluster_name.txt', delimiter=' ', header=None, names=['alias','full_name'])
    try:
        full_name = df[df['alias']==file_name]['full_name'].to_numpy()[0]
    except:
        print(file_name)
    exclusion = pd.read_csv('/home/yjm/miniconda/EyeCheck/id_excluded.txt', comment='#', delimiter=' ', header=None, names=['id', 'cluster', 'flag'])
    id_excluded = exclusion['id'].to_numpy().astype('int')
    cluster_excluded = exclusion['cluster'].to_numpy()
    count=0
    mass=[]
    sfr=[]
    mag=[]
    mass2 = np.logspace(5, 13, 100)
    mass2_sorted = np.sort(mass2)
    log10_sfr_ms0 = MS(mass2_sorted, z=0)
    log10_sfr_ms1 = MS(mass2_sorted, z=3)
    lists = glob.glob('*master.fits')
    if len(lists)==0:
        lists = glob.glob('*full_table.fits')
    hdulist2 = fits.open(lists[0])
    data3=hdulist2[1].data
    mu=data3['mu']
    try:
        f = glob.glob('*nuv.txt')[0]
    except:
        print('no nuv file of '+name)
        return 1
    B = pd.read_csv(f, comment='#',sep='\t',header=None,names=['index','ID','M_dust','ra','dec','M_16',
                                                                'M_84', 'L_IR','L_16',
                                                                'L_84', 'nuv','nuv_err','position',
                                                                'sfr_eazy','Av','M_stellar',
                                                                'z','z_16','z_84', 'beta_median', 'T_median',
                                                                'mu', 'm_16', 'm_84', 'sfr_eazy_16',
                                                                'sfr_eazy_84', 'Av_16', 'Av_84', 'flux_1.2mm',
                                                                'flux_1.2mm_err', 'sep', 'hband_snr'])
    B = B.iloc[1:]
    id_alma = B['ID'].to_numpy().astype('float64')
    cluster_alma = f.split('_')[0]

    coor=[]
    for i in range(id_alma.size):
        if (id_alma[i] in id_excluded and cluster_alma in cluster_excluded):
            count+=1
            continue
        else:
            coor = np.append(coor, i)
    if len(coor)!=0:
        coor = coor.astype('int')
    else:
        coor = range(id_alma.size)
    # to apply coor: Mdust_errors[:,coor]
    SFR_IR = 1.09e-10 * B['L_IR'].to_numpy().astype(float)
    L_IR = B['L_IR'].to_numpy().astype(float)
    M_dust = B['M_dust'].to_numpy().astype(float)
    Av = B['Av'].to_numpy().astype(float)
    Av_16 = B['Av_16'].to_numpy().astype(float)
    Av_84 = B['Av_84'].to_numpy().astype(float)
    M_stellar = B['M_stellar'].to_numpy().astype(float)
    flux_alma = B['flux_1.2mm'].to_numpy().astype(float)
    flux_alma_err = B['flux_1.2mm_err'].to_numpy().astype(float)
    ra = B['ra'].to_numpy().astype(float)
    dec = B['dec'].to_numpy().astype(float)
    SFR_eazy = B['sfr_eazy'].to_numpy().astype(float)
    mu = B['mu'].to_numpy().astype(float)
    M_gas = 100*M_dust
    T_dust = B['T_median'].to_numpy().astype(float)
    M_16 = B['M_16'].to_numpy().astype(float)
    M_84 = B['M_84'].to_numpy().astype(float)
    M_stellar_16 = B['m_16'].to_numpy().astype(float)
    M_stellar_84 = B['m_84'].to_numpy().astype(float)
    sfr_eazy_16 = B['sfr_eazy_16'].to_numpy().astype(float)
    sfr_eazy_84 = B['sfr_eazy_84'].to_numpy().astype(float)
    Sv_nuv = B['nuv'].to_numpy().astype(float) * 1e-29 # from uJy to erg/s/cm^2
    nuv_err = B['nuv_err'].to_numpy().astype(float) * 1e-29
    redshift = B['z'].to_numpy().astype(float)
    sep = B['sep'].to_numpy().astype(float)
    f160w_snr = B['hband_snr'].to_numpy().astype(float)
    SFR_UV, L_UV = nuv_to_sfr(Sv_nuv, redshift)
    _, L_UV_err = nuv_to_sfr(nuv_err, redshift)
    L_16 = B['L_16'].to_numpy()
    L_84 = B['L_84'].to_numpy()
    
    SFR = SFR_UV + SFR_IR

    flag_position = B['position'].to_numpy().astype('int')
    colors = np.where(flag_position==0,'blue',np.where(flag_position==1,'orange','red'))
    '''
    lists = glob.glob('*full_table.fits')
    if len(lists)==0:
        lists = glob.glob('*master.fits')
    fulltable = Table.read(lists[0])
    mass_full = fulltable['mass']
    try:
        sfr_full = fulltable['sfr']
    except:
        sfr_full = fulltable['SFR']
    Av_full = fulltable['Av']
    z_full = fulltable['z_phot']

    fig2, axes2 = plt.subplot_mosaic([['1', '2']], figsize=(16,7))
    axes2 = list(axes2.values())

    sfr_total = sfr_calculate(B['nuv'], B['L_IR'].to_numpy(), redshift)
    axes2[0].plot(np.log10(mass2_sorted)-0.23, log10_sfr_ms0, label='main sequence z=0', c='tomato')
    axes2[0].fill_between(np.log10(mass2_sorted)-0.23, log10_sfr_ms0-0.15, log10_sfr_ms0+0.15, facecolor='gainsboro')
    axes2[0].plot(np.log10(mass2_sorted)-0.23, log10_sfr_ms1, label='main sequence z=3', c='royalblue')
    axes2[0].fill_between(np.log10(mass2_sorted)-0.23, log10_sfr_ms1-0.15, log10_sfr_ms1+0.15, facecolor='gainsboro')
    sc = axes2[0].scatter(np.log10(mass_full), np.log10(sfr_full), s=10, c=z_full)
    axes2[0].scatter(np.log10(B['M_stellar']), np.log10(SFR), s=60, marker='x', c='red')
    axes2[0].set_xlabel(r'$log_{10}( M_*$/1e9 [$M_{\odot}$] )')
    axes2[0].set_ylabel(r'$log_{10} SFR$', labelpad=5)
    axes2[0].grid()

    plt.close(fig2)
    fig2.savefig('./pics/'+name+'_main sequence.png')
    '''
    files_name = np.ones(id_alma.size).astype('str')
    files_name[:] = file_name
    full_name1 = np.ones(id_alma.size).astype('str')
    full_name1[:] = full_name
    return (ra, dec, SFR_IR, SFR_UV, L_IR, L_UV, L_UV_err, M_dust, Av, SFR_eazy, M_stellar, M_gas, M_16, M_84,
            redshift, L_16, L_84, colors, files_name, id_alma, mu, M_stellar_16, M_stellar_84,
            sfr_eazy_16, sfr_eazy_84, Av_16, Av_84, T_dust, full_name1, flux_alma, flux_alma_err,
            sep, f160w_snr)