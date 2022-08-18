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
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from matplotlib.offsetbox import AnchoredText
import os
import scipy.special as sc
import gc
import warnings
import eazy
import time
import grizli
from grizli.pipeline import photoz
import numpy as np
from astropy.table import Table

h = 6.62e-27 # erg/s
k_b = 1.38e-16 # erg/K
c = 3e10 # speed of light: cm/s

def Lv(freq, A, beta, Tdust):
    return A * freq**(3+beta) / (np.exp(h * freq / (k_b * Tdust ) ) - 1 )

def Lv_rescaled(freq, N, A, beta, Tdust):  # Lv
    return N * A * freq**(3+beta) / (np.exp(h * freq / (k_b * Tdust ) ) - 1 )

def T_dust(z):
    return 32.9 + 4.6*(z-2), 2.4+0.35*(z-2)  # mean value and dispersion

def trace(m, z, a0=1.5, a1=0.3, a2=2.5, m0=0.5, m1=0.36):
    r = np.log10(1 + z)
    M = np.log10(m/1e9)
    p = M-m1-a2*r
    p[p<0] = 0
    return M - m0 + a0*r - a1*p**2

def sfr_calculate(nuv_flux, L_IR, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    d_L = cosmo.luminosity_distance(z).to(u.cm)
    L_v = 4*3.14*d_L.value**2*nuv_flux * 1e-29  # from uJy to erg/s/cm^2
    L_UV = 1.5 * 3e8 / 2.8e-7 * L_v / 3.82e33
    SFR_UV = 2.2 * 1.09e-10 * L_UV
    SFR_IR = 1.09e-10 * L_IR
    return SFR_UV + SFR_IR

def main(name):
    #warnings.filterwarnings("ignore")
    path='/home/yjm/miniconda/Outputs'
    path2='/home/yjm/miniconda/PICS'
    if name!='r0032':
        return
    if name!='overall_UVJ.png':
        try:
            os.chdir(path+'/'+name)
        except:
            os.mkdir(path+'/'+name)
            os.chdir(path+'/'+name)
    else:
        return
    #images = glob.glob(path+'/'+name+'/pics/*.png')[-1].split('/')[-1]
    '''
    try:
        objects = glob.glob('*L_IR_nuv.txt')[0]
        mtime = os.path.getmtime(objects)
        time_difference = time.time() - mtime
        if time_difference < 1800:
            return
    except:
        print(name+' no nuv_txt')
    '''
    file = glob.glob('*_irac_phot.fits')[0]
    root=file.split('_')[0]
    try:
        params = glob.glob('*masked.eazypy.zphot.param')[0]
        translate = glob.glob('*masked.eazypy.zphot.translate')[0]
        zeropoint = glob.glob('*eazypy.zphot.zeropoint')[0]
        self = eazy.photoz.PhotoZ(param_file=params, 
                              translate_file=translate,
                              verbose=True,
                              zeropoint_file=zeropoint, 
                              load_prior=False, load_products=True)
    except Exception as e:
        print('\n '+name+' '+str(e) + '\n')
        return
        # Turn off error corrections derived above
    #self.set_sys_err(positive=True)

    # Full catalog
    #sample = np.isfinite(self.ZSPEC)

    # fit_parallel renamed to fit_catalog 14 May 2021
    #self.fit_catalog(self.idx[sample], n_proc=8, verbose=False)
    lamb_rest = (np.arange(8, 1200) * 1e-6  ) # micron meters to meters
    freq_rest = scc.c / lamb_rest # Hz
    freq_rest_8_1000 = scc.c / (np.arange(8, 1000) * 1e-6  )
    freq_sorted = np.sort(freq_rest)
    freq_8_1000_sorted = np.sort(freq_rest_8_1000)
    index_1000 = np.where(lamb_rest==np.min(abs(freq_sorted-1)))
    kappa0 = 1.3   # cm^2/g --> m^2/kg
    v0 = 6.66e11  #Hz
    M_16=[]
    M_84=[]
    M_50=[]
    beta_median=[]
    T_median=[]
    L_16 = []
    L_84 = []
    L_UV = []
    L_50 = []
    detected_fits = glob.glob('*alma_detected.fits')[0]
    table = Table.read(detected_fits)
    
    magnification = np.array(table['magnification'])
    lamb_alma = np.array(table['lamb_alma'])*1e-6 # micron meters --> meters
    Sv_obs = np.array(table['flux_1.2mm'])/magnification*1e-26   # from micron Jy to erg/sec/cm^2
    Sv_obs_err = np.array(table['flux_1.2mm_err'])/magnification*1e-26
    snr_alma = np.array(table['snr_1.2mm'])
    nuv_flux = np.array(table['NUV'])/magnification
    nuv_err = np.array(table['NUV_err'])/magnification
    id_hst  = np.array(table['id_hst']).astype('int')
    id_eazy = np.array(table['id_eazy']).astype('int')
    id_alma = np.array(table['id_alma']).astype('int')
    Av = np.array(table['Av'])
    z = np.array(table['z'])
    z_16 = np.array(table['z_16'])
    z_84 = np.array(table['z_84'])
    sep = np.array(table['separation'])
    ra = np.array(table['ra'])
    hband_snr = np.array(table['f160w_snr'])
    dec = np.array(table['dec'])
    M_stellar = np.array(table['mass'])/magnification
    M_stellar_16 = np.array(table['mass_16'])/magnification
    M_stellar_84 = np.array(table['mass_84'])/magnification
    sfr = np.array(table['sfr'])/magnification
    sfr_16 = np.array(table['sfr_eazy_16'])/magnification
    sfr_84 = np.array(table['sfr_eazy_84'])/magnification
    Av_16 = np.array(table['Av_16'])
    Av_84 = np.array(table['Av_84'])
    flag_position = np.array(table['flag_position'])
    freq_alma = scc.c / lamb_alma.astype('float64')
    try:
        os.chdir('./pics')
    except:
        os.mkdir('pics')
        os.chdir('./pics')
    coor=[]
    num_iteration=1000
    beta = np.random.normal(1.8, 0.2, size=(num_iteration,))
    #beta = np.ones(num_iteration)*1.5
    for j in range(id_alma.size):      # loop over galaxies
        fig, axes = plt.subplot_mosaic([['1','1','2','2','2'],
                                    ['3','3','3','4','4'],
                                    ],
                                   empty_sentinel='X',constrained_layout=True, figsize=(15, 10))
        axes = list(axes.values())
        m_50=np.array([0.])
        l_50=np.array([0.])
        if z[j]<0:
            coor = np.append(coor, j)
            axes[0].text(0.2, 0.5, 'Error: redshift < 0')
            T = np.zeros(num_iteration)
            beta = np.zeros(num_iteration)
            T[500]=1
            median_position=np.where(T==T[500])
            M_16 = np.append(M_16, 0.)
            M_50 = np.append(M_50, 0.)
            M_84 = np.append(M_84, 0.)
            L_16 = np.append(L_16, 0.)
            L_50 = np.append(L_50, 0.)
            L_84 = np.append(L_84, 0.)
           
        elif Sv_obs[j]<0:
            coor = np.append(coor, j)
            axes[0].text(0.2, 0.5, 'Error: flux at 1.2mm < 0')
            T = np.zeros(num_iteration)
            beta = np.zeros(num_iteration)
            T[500]=1
            median_position=np.where(T==T[500])
            M_16 = np.append(M_16, 0.)
            M_50 = np.append(M_50, 0.)
            M_84 = np.append(M_84, 0.)
            L_16 = np.append(L_16, 0.)
            L_50 = np.append(L_50, 0.)
            L_84 = np.append(L_84, 0.)
        else:
            coor = np.append(coor, j)
            lamb_obs = lamb_rest * (1 + z[j])
            cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
            d_L = cosmo.luminosity_distance(z[j])
            d_L = d_L.to(u.cm)  # from parsec to cm
            
            low, high = T_dust(z[j])
            T = np.random.normal(low, high, size=(num_iteration,))
            if Sv_obs_err[j]>0:
                Sv_distribution = np.random.normal(Sv_obs[j], Sv_obs_err[j], size=(num_iteration,))
            else:
                Sv_distribution = np.ones(num_iteration)*Sv_obs[j]
            # T = np.ones(num_iteration)*35
            count=0
            T_range=[]
            M_dust=[]
            l_IR = []
            a = 2e-26
            for i in range(num_iteration):    # loop over iteration
                Lv_rest = 4*scc.pi*d_L**2*Sv_distribution[i]/(1+z[j])  # unit: erg/s/Hz
                N = Lv_rest.value / Lv(freq_alma[j], 2e-26, beta[i], T[i])
                Lv_rescale = N  * Lv(freq_rest, 2e-26, beta[i], T[i]) 
                Sv_rescale = Lv_rescale * (1 + z[j]) / (4 * np.pi * d_L**2)   #
                axes[0].plot(lamb_obs*1e6, Sv_rescale*1e26, zorder=0, c='lightgray')      # convert flux back to mJy
                axes[0].set_ylim(1e-4, 1e3)
                axes[0].set_yscale('log')
                axes[0].set_xscale('log')
                nu = 1.5e11 * (1+z[j])
                T_range = np.append(T_range, T[i])
                kappa = kappa0 * (nu / v0)**beta[i]    # m^2/kg
                # mistake here: Sv_rest and func() have different units
                L = Lv_rest.value / (Lv(freq_alma[j], a, beta[i], T[i]) )  * Lv(nu, a, beta[i], T[i])
                blackbody = 2 * h / c**2 * nu**3 / (np.exp(h*nu / (k_b*T[i])) - 1)
                mass = L / (4 * np.pi * kappa * blackbody )  # kg
                M_dust = np.append(M_dust, mass)
                #L_integrated = np.sum(Lv_rescale[0:-1]) * dx[-1]
                sol=quad(Lv_rescaled, freq_rest_8_1000[-1], freq_rest_8_1000[0], args=(N, a, beta[i], T[i]))
                L_integrated = sol[0]
                l_IR = np.append(l_IR, L_integrated/3.82e33 ) # in unit of L_sun
            m_16 = np.percentile(M_dust, 16, method='nearest')
            M_16 = np.append(M_16, m_16)
            m_84 = np.percentile(M_dust, 84, method='nearest')
            M_84 = np.append(M_84, m_84)
            l_16 = np.percentile(l_IR, 16, method='nearest')
            L_16 = np.append(L_16, l_16)
            l_50 = np.percentile(l_IR, 50, method='nearest')
            L_50 = np.append(L_50, l_50)
            l_84 = np.percentile(l_IR, 84, method='nearest')
            L_84 = np.append(L_84, l_84)
            m_50 = np.percentile(M_dust, 50, method='nearest')
            M_50 = np.append(M_50, m_50)

            median_position = np.where(M_dust==m_50)
            Lv_rescale = Lv_rest.value / Lv(freq_alma[j], 2e-26, beta[median_position], T[median_position])  * Lv(freq_rest, 2e-26, beta[median_position], T[median_position])
            Sv_rescale = Lv_rescale * (1 + z[j]) / (4 * np.pi * d_L**2)
            axes[0].plot(lamb_obs*1e6, Sv_rescale*1e26, zorder=0, c='black')
            # scatter
            axes[0].scatter(1200 ,Sv_obs[j]*1e26, zorder=10)
            if Sv_obs_err[j]>0:
                axes[0].errorbar(1200 ,Sv_obs[j]*1e26, yerr=Sv_obs_err[j]*1e26, capsize=5, c='grey', zorder=9)
            #ax.set_xlim(0, 3e13)
            axes[0].set_xlabel(r'$\lambda_{obs} / \mu m $')
            axes[0].set_ylabel(r'$Sv_{obs} / mJy $')
            M_dust = np.sort(M_dust)/2e33
        
        if flag_position[j]==0:
            position='foreground'
        elif flag_position[j]==1:
            position='inside cluster'
        else:
            position='background'
        try:
            m_exp = int(np.log10((m_50/2e33) // 1))
            m_base = m_50/2e33 / 10**m_exp
            m_base_error1 = (m_50 - m_16)/2e33 / 10**m_exp
            m_base_error2 = (m_84 - m_50)/2e33 / 10**m_exp

            l_exp = int(np.log10(l_50) // 1)
            l_base = l_50 / 10**l_exp
            l_base_error1 = (l_50 - l_16) / 10**l_exp
            l_base_error2 = (l_84 - l_50) / 10**l_exp
        except:
            m_exp = 0
            l_exp = 0
            m_base = m_base_error1 = m_base_error2 = 1
            l_base = l_base_error1 = l_base_error2 = 1
            
        string = ('ALMA ID=' + name + '_' + str(id_alma[j]) + '\n'+
              'RA=' + str('{:.2f}'.format(ra[j]))  + '\n'+
              'DEC='+ str('{:.2f}'.format(dec[j]))  + '\n' +
              'z=' + str('{:.2f}'.format(z[j])) + '\n' +  
              'position:' + position + '\n' +
              r'$\lambda_{\mathrm{rest}}$=' + str('{:.2f}'.format(lamb_alma[j]*1e6)) + r'$\mu \mathrm{m}$' '\n' +
              'HST ID='+ root + '_' + str(int(id_eazy[j])) + '\n'  +
              'sep=' + str('{:.2f}'.format(sep[j])) + 'arcsec'  + '\n' +
              'SNR=' + str(snr_alma[j]) + '\n' +
              r'$\mathrm{M_{dust}}=$' + r'${:.2f}^{{\mathrm{{+}}{:.2f}}}_{{{:.2f}}} \times 10^{{{}}}$'.format(m_base, m_base_error2, -m_base_error1, m_exp) + r'$\mathrm{M_{\odot}}$'+'\n' + 
              r'$M_{\mathrm{stellar}}$=' + str('{:.2e}'.format(M_stellar[j])) + r'$\mathrm{M_{\odot}}$'+ '\n' +
              r'$\mathrm{L_{IR}}=$' + r'${:.2f}^{{\mathrm{{+}}{:.2f}}}_{{{:.2f}}} \times 10^{{{}}}$'.format(l_base, l_base_error2, -l_base_error1, l_exp) +  r'$\mathrm{L_{\odot}}$'+ '\n'
              r'Av=' + str('{:.2f}'.format(Av[j])) + '\n' +
              r'$T_{\mathrm{median}}$=' + str('{:.2f}'.format(T[median_position][0])) + 'K \n' +
              r'$\beta_{\mathrm{median}}=$' + str('{:.2f}'.format(beta[median_position][0]))
             )
        beta_median.append(beta[median_position])
        T_median.append(T[median_position])
        axes[1].text(0.3, 0.1, string, fontsize=20)
        axes[1].axis('off')
        # SED plot and p(z)
        b = list([axes[2],axes[3]])
        
        c1 = SkyCoord(ra[j]*u.deg, dec[j]*u.deg, frame='fk5')
        c2 = SkyCoord(self.cat['ra'].value*u.deg, self.cat['dec'].value*u.deg, frame='fk5')
        sep1 = c2.separation(c1)
        coor_match = np.argmin(sep1)
        
        _, data = self.show_fit(self.cat['id'][coor_match], show_fnu=0,figsize=(10,5), axes=b)

        axes[2].set_ylim(-1, np.max(data['model'] + data['efobs'])+3)
        axes[2].set_ylabel(r'$f_{\lambda} ]$ / [mJy]')
        axes[2].set_xlabel(r'$\lambda_{obs} \ / \ [\mu $m]')
        xlim=[0.3,9]
        axes[2].set_xlim(xlim)
        axes[2].semilogx()
        xt = np.array([0.1, 0.5, 1, 2, 4, 8, 24, 160, 500])*1e4
        valid_ticks = (xt > xlim[0]*1.e4) & (xt < xlim[1]*1.e4)
        if valid_ticks.sum() > 0:
            xt = xt[valid_ticks]
            axes[2].set_xticks(xt/1.e4)
            axes[2].set_xticklabels(xt/1.e4)
        axes[2].grid()
        
        ix = self.idx[self.OBJID == self.cat['id'][coor_match]][0]
        txt = '{0}\nID={1}'
        txt = txt.format(self.param.params['MAIN_OUTPUT_FILE'], 
                                         self.OBJID[ix])
        axes[2].text(0.95, 0.95, txt, ha='right', va='top', fontsize=9,
                transform=axes[2].transAxes, 
                bbox=dict(facecolor='w', alpha=0.5), zorder=10)
        ymax = np.nanpercentile((data['efobs'] + data['model']), 95) # from EAZY
        try:
            axes[2].set_ylim(-0.1*ymax, 1.2*ymax)
        except:
            pass

        txt1 = 'z={0:.2f}'
        txt1 = txt1.format(data['z'])
        axes[3].text(0.85, 0.95, txt1, ha='left', va='top', fontsize=9,
                transform=axes[3].transAxes, 
                bbox=dict(facecolor='w', alpha=0.5), zorder=10)
        axes[3].set_ylabel('p(z)')
        axes[3].set_xlabel('z')
        axes[3].grid()
        
        plt.close(fig)
        os.chdir(path2+'/'+name)
        fig.savefig(str(id_alma[j])+'_'+name+'.pdf')
        gc.collect() 

    np.set_printoptions(precision=2)
    if len(coor)!=0:
        coor = np.array(coor).astype('int')
    if len(coor)==1:
        coor = coor[0]
    difference16 = M_50 - M_16
    difference84 = M_84 - M_50
    os.chdir('../')
    beta_median = np.array(beta_median, dtype=object).ravel()
    T_median = np.array(T_median, dtype=object).ravel()
    
    os.chdir(path+'/'+name)
    with open(name+'_dustmass_L_IR_nuv.txt', 'w') as f:
        f.write('# column 0 = index \r\n')
        f.write('# column 1 = ID \r\n')
        f.write('# column 2 = M(50%) / [M_sun] \r\n')
        f.write('# column 3 = M(16%) / [M_sun] \r\n')
        f.write('# column 4 = M(84%) / [M_sun] \r\n')
        f.write('# column 5 = L(50%) / [L_sun] \r\n')
        f.write('# column 6 = L(16%) / [L_sun] \r\n')
        f.write('# column 7 = L(84%) / [L_sun] \r\n')
        f.write('# column 8 = nuv / uJy \r\n')
        f.write('# column 9 = position \r\n')
        f.write('# column 10 = star formation rate from EAZY \r\n')
        f.write('# column 11 = Av \r\n')
        f.write('# column 12 = stellar mass \r\n')
        f.write('# column 13 = redshift \r\n')
        f.write('# column 14 = median beta \r\n')
        f.write('# column 15 = median dust temperature \r\n')
        f.write('# flag of position: 0=foreground, 1=inside cluster, 2=background \r\n')
        if isinstance(id_alma, int):
            id_alma = np.array([id_alma])
            nuv_flux = np.array([nuv_flux])
            flag_position = np.array([flag_position])
            sfr = np.array([sfr])
            Av = np.array([Av])
            M_stellar = np.array([M_stellar])
            z = np.array([z])
        try:
            df = pd.DataFrame({'# ID': id_alma[coor], 'M_dust': M_50/2e33,'ra':ra[coor],'dec':dec[coor],
                       'M(16%)': M_16/2e33, 'M(84%)': M_84/2e33,
                       'L_IR':L_50 , 'L(16%)':L_16, 'L(84%)':L_84,
                       'nuv':nuv_flux[coor],'nuv_err':nuv_err[coor], 'position':flag_position[coor], 'sfr_eazy': sfr[coor],
                       'Av':Av[coor], 'M_stellar':M_stellar[coor], 'z':z[coor], 'z_16':z_16[coor],
                       'beta_median':beta_median, 'z_84':z_84[coor],
                       'T_median':T_median, 'magnification':magnification[coor], 'm_stellar_16':M_stellar_16[coor],
                       'm_stellar_84':M_stellar_84[coor], 'sfr_eazy_16':sfr_16[coor], 'sfr_eazy_84':sfr_84[coor],
                       'Av_16':Av_16[coor], 'Av_84':Av_84[coor], 'flux_1.2mm':np.array(table['flux_1.2mm'])[coor],
                       'flux_1.2_err':np.array(table['flux_1.2mm_err'])[coor], 'separation':sep[coor], 'hband_snr':hband_snr
                              })
        except Exception as e:
            print(e)
            print(name)
        df.to_csv(f, sep='\t')
    print(name + ' is done')
    
    
    