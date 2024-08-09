# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 01:11:50 2024

@author: mapor
"""

import numpy as np
import scipy.optimize as opt

from drift_layer_class import drift_layer

import pathos.multiprocessing as mp

from functools import partial

import tqdm
import time
import platform
import sys

import matplotlib.pyplot as plt

class psw_minimization_direct:
    def __init__(self,materialparams,num_procs):
        self.material = materialparams
        self.num_procs = num_procs
        #self.opt_target = drift_layer(materialparams,drift_type='const',
        #                              drift_doping_params=[1e12,0.1])
        self.__itnum__ = 0
    
    def __constrained_psw__(self,x,opt_target,vbr,alpha):
        self.opt_target.dop_params = [10**x[0],x[1]]
        self.opt_target.quad_find_vbr_ionint(1)
        mu = 10
        #print(x)
        #print(self.opt_target.vbr)
        #print(self.opt_target.sw_fom1(self.opt_target.vbr, alpha))
        #print(self.opt_target.__solve_w__(self.opt_target.vbr))
        return self.opt_target.sw_fom1(self.opt_target.vbr, alpha)+x[2]*(self.opt_target.vbr-vbr)+0.5*mu*(self.opt_target.vbr-vbr)**2
    
    def __unconstrained_psw__(self,lognd,vbr,alpha,material_params):
        opt_target = drift_layer(material_params,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        self.__find_t__(opt_target,vbr, lognd)
        #self.opt_target.dop_params = [10**lognd,t]
        #self.opt_target.quad_find_vbr_ionint(1)
        #print(opt_target.vbr)
        #print(opt_target.sw_fom1(opt_target.vbr, alpha))
        #t2 = time.time()
        #print(f'elapsed time: {t2-t1:.2f}')
        return opt_target.sw_fom1(opt_target.vbr, alpha)
    
    def __unconstrained_ronsp__(self,lognd,vbr,material_params):
        opt_target = drift_layer(material_params,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        self.__find_t__(opt_target,vbr, lognd)
        #self.opt_target.dop_params = [10**lognd,t]
        #self.opt_target.quad_find_vbr_ionint(1)
        #print(opt_target.vbr)
        #print(opt_target.sw_fom1(opt_target.vbr, alpha))
        #t2 = time.time()
        #print(f'elapsed time: {t2-t1:.2f}')
        return opt_target.ron_sp(opt_target.vbr)
    
    def __find_t__(self,opt_target,vbr,lognd):
        #print(f'finding t for {10**lognd:.6e}')
        def resfunc(logt):
            opt_target.dop_params = [10**lognd,10**logt]
            #opt_target.quad_find_vbr_ionint(1)
            opt_target.bracket_find_vbr_ionint(1)
            #quad_find_vbr_ionint(1)
            #print(10**logt)
            #print(opt_target.vbr)
            return opt_target.vbr - vbr
        conv = False
        
        bmin = -5
        while not conv:
            try:
                res = opt.root_scalar(resfunc,method='brentq',bracket=[bmin,1],xtol=1e-9)
                conv = True
            except:
                bmin = bmin-0.1
                continue 
        return 10**res.root
    
    #def __find_t__(self,opt_target,vbr,lognd):
    #    print(f'finding t for {10**lognd:.2e}')
    #    def resfunc(logt):
    #        opt_target.dop_params = [10**lognd,10**logt]
    #        opt_target.quad_find_vbr_ionint(1)
    #        print(opt_target.vbr)
    #        return opt_target.vbr - vbr
    #    res = opt.root(resfunc,x0=-4.0,method='hybr',jac=False)
    #    return 10**res.x[0]
    
    def __find_nd__(self,opt_target,vbr,logt):
        #print('running nd search...')
        def resfunc(lognd):
            opt_target.dop_params = [10**lognd,10**logt]
            opt_target.bracket_find_vbr_ionint(1)
            #quad_find_vbr_ionint(1)
            #print(lognd)
            return opt_target.vbr - vbr
        conv = False
        
        bmax = 17
        while not conv:
            try:
                res = opt.root_scalar(resfunc,method='brentq',bracket=[13,bmax],xtol=1e-9)
                conv = True
            except:
                bmax = bmax+0.1
                #logt = logt + 0.25
                continue
        #print('finished nd search')
        return 10**res.root

    def __find_ndmax__(self,opt_target,vbr):
        return self.__find_nd__(opt_target, vbr, 0)         
            
    def find_minimum_fom(self,vbr,tol,alpha):
        opt_target = drift_layer(self.material,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        #t1 = time.time()
        lognmax = np.log10(self.__find_ndmax__(opt_target,vbr))
        #print(lognmax)
        #t2 = time.time()
        #print(f'elapsed time: {t2-t1:.2f}')
        tmax = self.__find_t__(opt_target,vbr, np.log10(0.99925*10**lognmax))
        #print(tmax)
        fomnpt = opt_target.sw_fom1(opt_target.vbr, alpha)
        ronspnpt = opt_target.ron_sp(opt_target.vbr)
        eossnpt = opt_target.eoss_sp(alpha*opt_target.vbr)
        
        res = opt.minimize_scalar(self.__unconstrained_psw__, bounds=(12,lognmax),
                                  args=(vbr,alpha,self.material), method='bounded')
        topt = self.__find_t__(opt_target,vbr, res.x)
        #t3 = time.time()
        #print(f'total elapsed time: {t3-t1:.2f}')
        ronspopt = opt_target.ron_sp(opt_target.vbr)
        eossopt = opt_target.eoss_sp(alpha*opt_target.vbr)
        
        return [[res.fun,ronspopt,eossopt],[10**res.x,topt],[fomnpt,ronspnpt,eossnpt],[10**lognmax,tmax]]     

    def find_minimum_ronsp(self,vbr,tol):
        opt_target = drift_layer(self.material,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        #t1 = time.time()
        lognmax = np.log10(self.__find_ndmax__(opt_target,vbr))
        #t2 = time.time()
        #print(f'elapsed time: {t2-t1:.2f}')
        tmax = self.__find_t__(opt_target,vbr, np.log10(0.99925*10**lognmax))
        ronspnpt = opt_target.ron_sp(opt_target.vbr)
        
        res = opt.minimize_scalar(self.__unconstrained_ronsp__, bounds=(12,lognmax),
                                  args=(vbr,self.material), method='bounded')
        topt = self.__find_t__(opt_target,vbr, res.x)
        #t3 = time.time()
        #print(f'total elapsed time: {t3-t1:.2f}')
        
        return [res.fun,[10**res.x,topt],ronspnpt,[10**lognmax,tmax]]     

    def __nd_fom_sw_wrapper__(self,lognd,vbr,alpha,material_params):
        opt_target = drift_layer(material_params,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        self.__find_t__(opt_target,vbr, lognd)

        #self.opt_target.dop_params = [10**lognd,t]
        #self.opt_target.quad_find_vbr_ionint(1)
        #print(opt_target.vbr)
        #print(opt_target.sw_fom1(opt_target.vbr, alpha))
        #t2 = time.time()
        #print(f'elapsed time: {t2-t1:.2f}')
        return [opt_target.sw_fom1(opt_target.vbr, alpha),opt_target.ron_sp(opt_target.vbr),
                opt_target.eoss_sp(alpha*opt_target.vbr),opt_target.n_actual(10**lognd)]

    def sweep_fom_vs_nd(self,vbr,ndmin,numruns,tol,alpha,savefile=True,scale_to_NPT=False,
                        filename=''):
        pool = mp.Pool(processes = num_procs)
        
        opt_target = drift_layer(self.material,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        
        t1 = time.time()
        lognmax = np.log10(self.__find_ndmax__(opt_target,vbr))
        
        objlist = np.log10(np.logspace(np.log10(ndmin),np.log10(0.9995*10**lognmax),numruns))
        
        t2 = time.time()
        #print(f'starting psw evaluation @{t2-t1}')
        func = partial(self.__nd_fom_sw_wrapper__,vbr=vbr,alpha=alpha,material_params=self.material)
        results = list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        #pool.map(func,objlist)
        #list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        #print(f'elapsed time:{time.time()-t1}')
        pool.close()
        pool.terminate()
        pool.join()
        if savefile:
            fileout = open(filename,'w')
            header = 'Nd (cm^-3),ron,sp (ohm*cm^2)'
            fileout.write(header)
            for i,n in enumerate(objlist):
                if scale_to_NPT:
                    lineout = [str(10**n),str(results[i][0]/results[-1][0]),str(results[i][1]/results[-1][1]),
                               str(results[i][2]/results[-1][2]),str(results[i][3]/results[-1][3])]
                else:
                    lineout = [str(10**n),str(results[i][0]),str(results[i][1]),
                               str(results[i][2]),str(results[i][3])]
                linestr = ','.join(lineout) + '\n'
                fileout.write(linestr)
            fileout.close()
        return [objlist,results]
    
    def sweep_ronsp_vs_nd(self,vbr,ndmin,numruns,tol,savefile=True,filename=''):
        pool = mp.Pool(processes = num_procs)
        
        opt_target = drift_layer(self.material,drift_type='const',
                                      drift_doping_params=[1e12,0.1])
        
        t1 = time.time()
        lognmax = np.log10(self.__find_ndmax__(opt_target,vbr))
        
        objlist = np.log10(np.logspace(np.log10(ndmin),np.log10(0.9995*10**lognmax),numruns))
        
        t2 = time.time()
        #print(f'starting psw evaluation @{t2-t1}')
        func = partial(self.__unconstrained_ronsp__,vbr=vbr,material_params=self.material)
        results = list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        #pool.map(func,objlist)
        #list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        #print(f'elapsed time:{time.time()-t1}')
        pool.close()
        pool.terminate()
        pool.join()
        if savefile:
            fileout = open(filename,'w')
            header = 'Nd (cm^-3),ron,sp (ohm*cm^2)'
            fileout.write(header)
            for i,n in enumerate(objlist):
                lineout = [str(n),str(results[i])]
                linestr = ','.join(lineout) + '\n'
                fileout.write(linestr)
            fileout.close()
        return [objlist,results]
    
    def sweep_fom_min_vs_vbr(self,vbrmin,vbrmax,numruns,tol,alpha,savefile=False,filename=''):
        pool = mp.Pool(processes = num_procs)
        
        objlist = np.logspace(np.log10(vbrmin),np.log10(vbrmax),numruns)
        
        func = partial(self.find_minimum_fom, tol=tol, alpha=alpha)
        results = list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        
        pool.close()
        pool.terminate()
        pool.join()
        if savefile:
            fileout = open(filename,'w')
            header = 'vbr (V),sw_fom (V^2/Hz),ronsp_fom (ohm*cm^2),eoss_fom (J/cm^2), nd_fom (cm^-3),t_fom (cm),\
                sw_fom_npt (V^2/Hz),ronsp_npt (ohm*cm^2),eoss_npt (J/cm^2),nd_npt (cm^-3),t_npt (cm)\n'
            fileout.write(header)
            for i,v in enumerate(objlist):
                lineout = [str(v),str(results[i][0][0]),str(results[i][0][1]),str(results[i][0][2]),
                           str(results[i][1][0]),str(results[i][1][1]),
                           str(results[i][2][0]),str(results[i][2][1]),str(results[i][2][2]),
                           str(results[i][3][0]),str(results[i][3][1])]
                linestr = ','.join(lineout) + '\n'
                fileout.write(linestr)
            fileout.close()
                
        return [objlist,results]
    
    def sweep_ron_min_vs_vbr(self,vbrmin,vbrmax,numruns,tol,savefile=False,filename=''):
        pool = mp.Pool(processes = num_procs)
        
        objlist = np.logspace(np.log10(vbrmin),np.log10(vbrmax),numruns)
        
        func = partial(self.find_minimum_ronsp,tol=tol)
        results = list(tqdm.tqdm(pool.imap(func,objlist),total=len(objlist)))
        
        pool.close()
        pool.terminate()
        pool.join()
        
        if savefile:
            fileout = open(filename,'w')
            header = 'vbr (V),ron_sp_opt (ohm*cm^2),nd_fom (cm^-3),t_fom (cm),\
                ron_sp_npt (ohm*cm^2),nd_npt (cm^-3),t_npt (cm)\n'
            fileout.write(header)
            for i,v in enumerate(objlist):
                lineout = [str(v),str(results[i][0]),str(results[i][1][0]),
                           str(results[i][1][1]),str(results[i][2]),str(results[i][3][0]),
                           str(results[i][3][1])]
                linestr = ','.join(lineout) + '\n'
                fileout.write(linestr)
            fileout.close()
            
        return [objlist,results]
    
if __name__=='__main__':
    if platform.system() == 'Windows':
        mp.freeze_support()
        __spec__ = None
    
    def mun_si(nd,T):
        nf = 1.072e17
        #alpha=2.0
        #beta=0.7
        #gam=1.0
        mu_max=1429.23
        mu_min=55.24
        k2=(T/300)**-2.3
        k3=(T/300)**-3.8
        return mu_min+(k2*mu_max-mu_min)/(1+k3*(nd/nf)**0.73)
    
    def mun_sic(nd,T):
        nf = 1.94e17
        #alpha=2.0
        #beta=0.7
        #gam=1.0
        mu_max=950.0
        mu_min=27.87
        k2=(T/300)**-1.8
        k3=(T/300)**0.61
        return mu_min+(k2*mu_max-mu_min)/(1+k3*(nd/nf))
    
    def mun_gan(nd,T):
        nf = 2e17
        #alpha=2.0
        #beta=0.7
        #gam=1.0
        mu_max=1.0e3
        mu_min=55.0
        bi = (mu_min+mu_max*(nf/nd))/(mu_min-mu_max)
        k1=(T/300)**0.7
        k2=(T/300)**2.7
        return mu_max*(k1*bi/(1+k2*bi))
    
    def mun_gan_fmct(nd,T):
        nf = 1e17
        mu_max=1460.7
        mu_min=295.0
        alpha=0.66
        beta1=-1.02
        beta2=-3.84
        beta3=3.02
        beta4=0.81
        
        k1=(T/300)**beta1
        k2=(T/300)**beta2
        k3=(T/300)**beta3
        k4=(T/300)**beta4
        
        alpha_p=alpha*k4
        
        return mu_min*k1+(mu_max-mu_min)*k2/(1+(nd/(nf*k3))**alpha_p)
    
    #To the best of our knowledge, mun of b-Ga2O3 is effectively isotropic
    def mun_ga2o3_x(nd,T):
        a=56.0
        b=508.0
        c=278.0
        d=0.68
        nf=2.8e16
        return (a*(np.exp(b/T)-1)/(1+(nd/nf)*(1/(T-c)))**d)
    
    def mun_ga2o3_y(nd,T):
        a=56.0
        b=508.0
        c=278.0
        d=0.68
        nf=2.8e16
        return (a*(np.exp(b/T)-1)/(1+(nd/nf)*(1/(T-c)))**d)
    
    def mun_ga2o3_z(nd,T):
        a=56.0
        b=508.0
        c=278.0
        d=0.68
        nf=2.8e16
        return (a*(np.exp(b/T)-1)/(1+(nd/nf)*(1/(T-c)))**d)
    
    def mun_algan_20(nd,T):
        nf = 1e17
        mu_max=1401.3
        mu_min=312.1
        alpha=0.74
        beta1=-6.51
        beta2=-2.31
        beta3=7.07
        beta4=-0.86
        
        k1=(T/300)**beta1
        k2=(T/300)**beta2
        k3=(T/300)**beta3
        k4=(T/300)**beta4
        
        alpha_p=alpha*k4
        
        return mu_min*k1+(mu_max-mu_min)*k2/(1+(nd/(nf*k3))**alpha_p)
    
    def mun_algan_50(nd,T):
        nf = 1e17
        mu_max=1215.4
        mu_min=299.4
        alpha=0.8
        beta1=-5.7
        beta2=-2.29
        beta3=7.57
        beta4=-1.08
        
        k1=(T/300)**beta1
        k2=(T/300)**beta2
        k3=(T/300)**beta3
        k4=(T/300)**beta4
        
        alpha_p=alpha*k4
        
        return mu_min*k1+(mu_max-mu_min)*k2/(1+(nd/(nf*k3))**alpha_p)
    
    def mun_algan_80(nd,T):
        nf = 1e17
        mu_max=881.1
        mu_min=321.7
        alpha=1.01
        beta1=-1.60
        beta2=-3.69
        beta3=3.31
        beta4=0.44
        
        k1=(T/300)**beta1
        k2=(T/300)**beta2
        k3=(T/300)**beta3
        k4=(T/300)**beta4
        
        alpha_p=alpha*k4
        
        return mu_min*k1+(mu_max-mu_min)*k2/(1+(nd/(nf*k3))**alpha_p)
    
    def mun_aln(nd,T):
        nf = 1e17
        mu_max=683.8
        mu_min=297.8
        alpha=1.16
        beta1=-1.82
        beta2=-3.43
        beta3=3.78
        beta4=-0.86
        
        k1=(T/300)**beta1
        k2=(T/300)**beta2
        k3=(T/300)**beta3
        k4=(T/300)**beta4
        
        alpha_p=alpha*k4
        
        return mu_min*k1+(mu_max-mu_min)*k2/(1+(nd/(nf*k3))**alpha_p)
    
    def mup_diamond(nd,T):
        beta1 = 3.11
        nbeta = 4.1e18
        gammab = 0.617
        
        nf = 3.25e17
        mu_max=2016.0
        mu_min=0
        gammamu=0.73
        
        mu300=mu_min+(mu_max-mu_min)/(1+(nd/nf)**gammamu)
        beta=-1*beta1/(1+(nd/nbeta)**gammab)
        
        k1 = (T/300)**beta
        return mu300*k1
    
    def mun_cbn(nd,T):
        
        nf = 7.5e17
        mu_max=1464.7
        mu_min=0
        gammamu=0.825
        
        mu300=mu_min+(mu_max-mu_min)/(1+(nd/nf)**gammamu)
        
        return mu300
    
    Si_TB_n_props = [1.12,67.5*1e-8,2.51,0.106,49.4*1e-3]
    Si_TB_p_props = [1.12,25.1*1e-8,3.06,0.021,12.0*1e-3]
    
    siparams = {'temp':300.0,
                 'eps': 11.7,
                  'eg': 1.124,
                  'nc': 2.86e19,
                  'nv': 3.10e19,
                  'ii_model':'thornber',
                  'ii_params_n':Si_TB_n_props,
                  'ii_params_p':Si_TB_p_props,
                  'mun_model':mun_si,
                  'ea':0.045}
    
    sic_OC_n_props = lambda T: [1.43e5,0,4.93e6,2.37]
    sic_OC_p_props = lambda T: [3.14e6*(1+6.30e-3*(T-300)),0,1.18e7*(1+1.23e-3*(T-300)),1.02]
    
    sicparams = {'temp':300.0,
                  'eps': 9.7,
                   'eg': 3.23,
                   'nc': 1.66e19,
                   'nv': 3.3e19,
                  'ii_model':'okuto_crowell',
                  'ii_params_n':sic_OC_n_props(300),
                  'ii_params_p':sic_OC_p_props(300),
                  'mun_model': mun_sic,
                  'ea':0.055}
    
    GaN_chyn_ji_n_props = lambda T: [2.11e9,3.689e7]
    GaN_chyn_ji_p_props = lambda T: [4.39e6,1.8e7]
    
    GaN_chyn_cao_n_props = lambda T: [2.77e8*(1+3.09e-3*(T-298)),3.20e7*(1+4.03e-4*(T-298))]
    GaN_chyn_cao_p_props = lambda T: [8.53e6*(1+3.23e-3*(T-298)),1.48e7*(1+7.02e-4*(T-298))]
    
    GaN_chyn_maeda_n_props = lambda T: [2.69e7*(1+2e-3*(T-298)),2.27e7*(1+5e-4*(T-298))]
    GaN_chyn_maeda_p_props = lambda T: [4.32e6*(1+2e-3*(T-298)),1.31e7*(1+9e-4*(T-298))]
    
    GaNparams_ji = {'temp':300,
                 'eps': 8.9,
                  'eg': 3.43,
                  'nc': 2.24e18,
                  'nv': 2.51e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':GaN_chyn_ji_n_props(300),
                  'ii_params_p':GaN_chyn_ji_p_props(300),
                  'mun_model':mun_gan_fmct,
                  'ea':0.012}
    
    GaNparams_cao = {'temp':300,
                 'eps': 8.9,
                  'eg': 3.43,
                  'nc': 2.24e18,
                  'nv': 2.51e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':GaN_chyn_cao_n_props(300),
                  'ii_params_p':GaN_chyn_cao_p_props(300),
                  'mun_model':mun_gan_fmct,
                  'ea':0.012}
    
    GaNparams_maeda = {'temp':300,
                 'eps': 8.9,
                  'eg': 3.43,
                  'nc': 2.24e18,
                  'nv': 2.51e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':GaN_chyn_maeda_n_props(300),
                  'ii_params_p':GaN_chyn_maeda_p_props(300),
                  'mun_model':mun_gan_fmct,
                  'ea':0.012}
    
    def eff_DOS(mb):
        return 2*(2*np.pi*mb*0.517e6*0.0259/(1.24e-4)**2)**1.5
    
    ga2o3_x_chyn_n_props = [0.79e6,2.92e7]
    ga2o3_x_chyn_p_props = ga2o3_x_chyn_n_props
    
    ga2o3_y_chyn_n_props = [2.16e6,1.77e7]
    ga2o3_y_chyn_p_props = ga2o3_y_chyn_n_props
    
    ga2o3_z_chyn_n_props = [0.706e6,2.10e7]
    ga2o3_z_chyn_p_props = ga2o3_z_chyn_n_props
    
    ga2o3_exp_chyn_n_props = [2.16e6,1.77e7]
    ga2o3_exp_chyn_p_props = [5.75e6,1.77e7]
    
    ga2o3_x_params = {'temp':300,
                 'eps': 10.0,
                  'eg': 4.84,
                  'nc':3.718e18,
                  'nv':5.67e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':ga2o3_x_chyn_n_props,
                  'ii_params_p':ga2o3_x_chyn_p_props,
                  'mun_model':mun_ga2o3_x,
                  'ea':0.030}
    
    ga2o3_y_params = {'temp':300,
                 'eps': 10.0,
                  'eg':4.84,
                  'nc':3.718e18,
                  'nv':5.67e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':ga2o3_y_chyn_n_props,
                  'ii_params_p':ga2o3_y_chyn_p_props,
                  'mun_model':mun_ga2o3_y,
                  'ea':0.030}
    
    ga2o3_z_params = {'temp':300,
                 'eps': 10.0,
                  'eg':4.84,
                  'nc':3.718e18,
                  'nv':5.67e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':ga2o3_z_chyn_n_props,
                  'ii_params_p':ga2o3_z_chyn_p_props,
                  'mun_model':mun_ga2o3_z,
                  'ea':0.030}
    
    ga2o3_exp_params = {'temp':300,
                 'eps': 10.0,
                  'eg':4.84,
                  'nc':3.718e18,
                  'nv':5.67e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':ga2o3_exp_chyn_n_props,
                  'ii_params_p':ga2o3_exp_chyn_p_props,
                  'mun_model':mun_ga2o3_z,
                  'ea':0.030}
    
    eggan = lambda T: 3.507-(0.909e-3*T*T)/(T+830.0)
    egaln = lambda T: 6.23-(1.799e-3*T*T)/(T+1462.0)
    egalgan = lambda x,T: egaln(T)*x+eggan(T)*(1-x)-1.3*x*(1-x) 
    
    ncalgan = lambda x: eff_DOS(0.314*x+0.2*(1-x))
    nvalgan = lambda x: eff_DOS(0.417*x+1.0*(1-x))
    
    AlGaN_20_chyn_n_props = [1.5126e7,2.389e7]
    AlGaN_20_chyn_p_props = AlGaN_20_chyn_n_props
    
    AlGaN_20_params = {'temp':300,
                 'eps': 8.9*(1-0.2)+8.5*0.2,
                  'eg':egalgan(0.2,300),
                  'nc':ncalgan(0.2),
                  'nv':nvalgan(0.2),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlGaN_20_chyn_n_props,
                  'ii_params_p':AlGaN_20_chyn_p_props,
                  'mun_model':mun_algan_20,
                  'ea':0.070}
    
    AlGaN_50_chyn_n_props = [1.91519e7,3.694e7]
    AlGaN_50_chyn_p_props = AlGaN_50_chyn_n_props
    
    AlGaN_50_params = {'temp':300,
                 'eps': 8.9*(1-0.5)+8.5*0.5,
                  'eg':egalgan(0.5,300),
                  'nc':ncalgan(0.5),
                  'nv':nvalgan(0.5),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlGaN_50_chyn_n_props,
                  'ii_params_p':AlGaN_50_chyn_p_props,
                  'mun_model':mun_algan_50,
                  'ea':0.070}
    
    AlGaN_80_chyn_n_props = [1.2993e7,3.634e7]
    AlGaN_80_chyn_p_props = AlGaN_80_chyn_n_props
    
    AlGaN_80_params = {'temp':300,
                 'eps': 8.9*(1-0.8)+8.5*0.8,
                  'eg':egalgan(0.2,300),
                  'nc':ncalgan(0.8),
                  'nv':nvalgan(0.8),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlGaN_80_chyn_n_props,
                  'ii_params_p':AlGaN_80_chyn_p_props,
                  'mun_model':mun_algan_80,
                  'ea':0.070}
    
    AlN_chyn_n_props = [8.875e6,3.659e7]
    AlN_chyn_p_props = AlN_chyn_n_props
    
    AlN_chyn_n_bellotti_props = [10**6.9541,3.7706e7]
    AlN_chyn_p_bellotti_props = [10**5.3488,3.1e7]
    
    AlNparams = {'temp':300,
                 'eps': 8.5,
                  'eg':egaln(300),
                  'nc':ncalgan(1.0),
                  'nv':nvalgan(1.0),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlN_chyn_n_props,
                  'ii_params_p':AlN_chyn_p_props,
                  'mun_model':mun_aln,
                  'ea':0.070}
    
    AlN_dx_params = {'temp':300,
                 'eps': 8.5,
                  'eg':egaln(300),
                  'nc':ncalgan(1.0),
                  'nv':nvalgan(1.0),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlN_chyn_n_props,
                  'ii_params_p':AlN_chyn_p_props,
                  'mun_model':mun_aln,
                  'ea':0.211}
    
    AlN_bel_params = {'temp':300,
                 'eps': 8.5,
                  'eg':egaln(300),
                  'nc':ncalgan(1.0),
                  'nv':nvalgan(1.0),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlN_chyn_n_bellotti_props,
                  'ii_params_p':AlN_chyn_p_bellotti_props,
                  'mun_model':mun_aln,
                  'ea':0.070}
    
    AlN_bel_dx_params = {'temp':300,
                 'eps': 8.5,
                  'eg':egaln(300),
                  'nc':ncalgan(1.0),
                  'nv':nvalgan(1.0),
                  'ii_model':'chynoweth',
                  'ii_params_n':AlN_chyn_n_bellotti_props,
                  'ii_params_p':AlN_chyn_p_bellotti_props,
                  'mun_model':mun_aln,
                  'ea':0.211}
    
    diamond_300_chyn_n_props = [3.7e6,58.0e6]
    diamond_300_chyn_p_props = [4.2e6,21.0e6]
    
    diamond_300_params = {'temp':300,
                 'eps': 5.5,
                  'eg': 5.5,
                  'nc':5e18,
                  'nv':1.8e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':diamond_300_chyn_n_props,
                  'ii_params_p':diamond_300_chyn_p_props,
                  'mun_model':mup_diamond,
                  'ea':0.382}
    
    diamond_600_chyn_n_props = [3.7e6,58.0e6]
    diamond_600_chyn_p_props = [4.2e6,21.0e6]
    
    diamond_600_params = {'temp':600,
                 'eps': 5.5,
                  'eg': 5.5,
                  'nc':5e18,
                  'nv':1.8e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':diamond_600_chyn_n_props,
                  'ii_params_p':diamond_600_chyn_p_props,
                  'mun_model':mup_diamond,
                  'ea':0.382}
    
    cbn_chyn_n_props = [10**6.0796,5.184e7]
    cbn_chyn_p_props = [10**6.5066,3.3102e7]
    
    cbn_params = {'temp':300,
                 'eps': 7.1,
                  'eg': 6.1,
                  'nc': 1.3e19,
                  'nv': 2.0e19,
                  'ii_model':'chynoweth',
                  'ii_params_n':cbn_chyn_n_props,
                  'ii_params_p':cbn_chyn_p_props,
                  'mun_model':mun_cbn,
                  'ea':0.24}
    
    num_procs = 12
    
    Si = psw_minimization_direct(siparams,num_procs)
    SiC = psw_minimization_direct(sicparams,num_procs)
    GaN_ji = psw_minimization_direct(GaNparams_ji,num_procs)
    GaN_cao = psw_minimization_direct(GaNparams_cao,num_procs)
    GaN_maeda = psw_minimization_direct(GaNparams_maeda,num_procs)
    bGa2O3_x = psw_minimization_direct(ga2o3_x_params,num_procs)
    bGa2O3_y = psw_minimization_direct(ga2o3_y_params,num_procs)
    bGa2O3_z = psw_minimization_direct(ga2o3_z_params,num_procs)
    bGa2O3_exp = psw_minimization_direct(ga2o3_exp_params,num_procs)
    algan_20 = psw_minimization_direct(AlGaN_20_params,num_procs)
    algan_50 = psw_minimization_direct(AlGaN_50_params,num_procs)
    algan_80 = psw_minimization_direct(AlGaN_80_params,num_procs)
    aln = psw_minimization_direct(AlNparams,num_procs)
    diamond300 = psw_minimization_direct(diamond_300_params,num_procs)
    diamond600 = psw_minimization_direct(diamond_600_params,num_procs)
    cbn = psw_minimization_direct(cbn_params, num_procs)
    aln_dx = psw_minimization_direct(AlN_dx_params, num_procs)
    
    aln_bel = psw_minimization_direct(AlN_bel_params, num_procs)
    aln_bel_dx = psw_minimization_direct(AlN_bel_dx_params, num_procs)
    
    ndsweep = np.logspace(14,18,num=1000)
    mun = np.zeros(len(ndsweep))
    
    mob_fileout = open('aln_mun_nd.csv','w')
    
    for i,n in enumerate(ndsweep):
        mun[i] = mun_aln(n,300)
        mob_fileout.write(f'{n:.6e},{mun[i]:.6e}\n')
    
    mob_fileout.close()
    
    
    
    #out1 = GaN_cao.sweep_fom_vs_nd(400, 10**16.2, 200, 1e-3, 0.5, savefile=True, filename='GaN_400_50_Nd.csv')
    # out2 = aln_bel.sweep_fom_vs_nd(400,1e17,200,1e-3,0.5,savefile=True,scale_to_NPT=False,
    #                                filename='aln_bel_50_400_Nd_w_n.csv')
    # out3 = aln_bel.sweep_fom_vs_nd(10000,1e15,200,1e-3,0.5,savefile=True,scale_to_NPT=False,
    #                                filename='aln_bel_50_10000_Nd_w_n.csv')
    # out4 = aln_bel_dx.sweep_fom_vs_nd(400,1e17,200,1e-3,0.5,savefile=True,scale_to_NPT=False,
    #                                filename='aln_bel_dx_50_400_Nd_w_n.csv')
    # out5 = aln_bel_dx.sweep_fom_vs_nd(10000,1e15,200,1e-3,0.5,savefile=True,scale_to_NPT=False,
    #                                filename='aln_bel_dx_50_10000_Nd_w_n.csv')
    
    
    #bGa2O3_x.find_minimum_fom(200.0, 1e-3, 1.0)
    #out = GaN_cao.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.25, savefile=True, filename='GaN_cao_25_FOM.csv')
    # out2 = GaN_cao.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='GaN_cao_50_FOM.csv')
    # #out3 = GaN_cao.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.75, savefile=True, filename='GaN_cao_75_FOM.csv')
    
    # out4 = Si.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='Si_50_FOM.csv')
    
    # out5 = SiC.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='SiC_50_FOM.csv')
    # out6 = bGa2O3_x.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='bGa2O3_x_50_FOM.csv')
    # #out7 = bGa2O3_y.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='bGa2O3_y_FOM.csv')
    # #out8 = bGa2O3_z.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='bGa2O3_z_FOM.csv')
    # out8 = bGa2O3_exp.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='bGa2O3_exp_50_FOM.csv')
    # #out9 = algan_20.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 1.0, savefile=True, filename='algan_20_FOM.csv')
    # #out10 = algan_50.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 1.0, savefile=True, filename='algan_50_FOM.csv')
    # #out11 = algan_80.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 1.0, savefile=True, filename='algan_80_FOM.csv')
    #out12 = diamond300.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='diamond_300_50_FOM.csv')
    #out12 = diamond600.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='diamond_600_50_FOM.csv')
    
    #out13 = cbn.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='cbn_50_FOM.csv')
    #out14 = aln.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='aln_50_FOM.csv')
    #out14 = aln_dx.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='aln_dx_50_FOM.csv')
    # out14 = aln_bel.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='aln_bel_50_FOM.csv')
    # out14 = aln_bel_dx.sweep_fom_min_vs_vbr(200.0, 20000.0, 100, 1e-3, 0.5, savefile=True, filename='aln_bel_dx_50_FOM.csv')
    
    
    #out7 = diamond300.sweep_fom_min_vs_vbr(3200.0, 3300.0, 2, 1e-3, 0.25, savefile=True, filename='diamond_300_3300.csv')
    
    # out5 = SiC.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='SiC_50_FOM_10000.csv')
    # out6 = GaN_cao.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='GaN_CaO_50_FOM_10000.csv')
    # out7 = bGa2O3_x.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='bGa2O3_x_50_FOM_10000.csv')
    # out8 = bGa2O3_exp.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='bGa2O3_exp_50_FOM_10000.csv')
    # out9 = diamond300.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='diamond_300_50_FOM_10000.csv')
    # out10 = diamond600.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='diamond_600_50_FOM_10000.csv')
    # out11 = aln_bel.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='aln_bel_50_FOM_10000.csv')
    # out12 = aln_bel_dx.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='aln_bel_dx_50_FOM_10000.csv')
    #out13 = cbn.sweep_fom_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, 0.5, savefile=True, filename='cbn_50_FOM_10000.csv')
    
    # out5 = SiC.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='SiC_50_FOM_3300.csv')
    # out6 = GaN_cao.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='GaN_CaO_50_FOM_3300.csv')
    # out7 = bGa2O3_x.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='bGa2O3_x_50_FOM_3300.csv')
    # out8 = bGa2O3_exp.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='bGa2O3_exp_50_FOM_3300.csv')
    # out9 = diamond300.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='diamond_300_50_FOM_3300.csv')
    # out10 = diamond600.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='diamond_600_50_FOM_3300.csv')
    # out11 = aln_bel.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='aln_bel_50_FOM_3300.csv')
    # out12 = aln_bel_dx.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='aln_bel_dx_50_FOM_3300.csv')
    #out13 = cbn.sweep_fom_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, 0.5, savefile=True, filename='cbn_50_FOM_3300.csv')
    
    #out2 = GaN_ji.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='GaN_Ji_Ronsp.csv')
    
    #out2 = Si.sweep_ron_min_vs_vbr(200.0, 10000.0, 100, 1e-3, savefile=True, filename='Si_Ronsp.csv')
    # out2 = GaN_cao.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='GaN_Cao_Ronsp.csv')
    # out2 = SiC.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='SiC_Ronsp.csv')
    # out2 = bGa2O3_x.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='bGa2O3_x_Ronsp.csv')
    # out2 = bGa2O3_exp.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='bGa2O3_exp_Ronsp.csv')
    # out2 = diamond300.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='diamond_300_Ronsp.csv')
    # out2 = diamond600.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='diamond_600_Ronsp.csv')
    # out2 = cbn.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='cbn_Ronsp.csv')
    # out2 = aln_bel.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='aln_bel_Ronsp.csv')
    # out2 = aln_bel_dx.sweep_ron_min_vs_vbr(200.0, 20000.0, 100, 1e-3, savefile=True, filename='aln_bel_dx_Ronsp.csv')
    
    #out2 = GaN_cao.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='GaN_Cao_Ronsp_3300.csv')
    #out2 = SiC.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='SiC_Ronsp_3300.csv')
    # out2 = bGa2O3_x.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='bGa2O3_x_Ronsp_3300.csv')
    # out2 = bGa2O3_exp.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='bGa2O3_exp_Ronsp_3300.csv')
    # out2 = diamond300.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='diamond_300_Ronsp_3300.csv')
    # out2 = diamond600.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='diamond_600_Ronsp_3300.csv')
    # out2 = cbn.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='cbn_Ronsp_3300.csv')
    # out2 = aln_bel.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='aln_bel_Ronsp_3300.csv')
    # out2 = aln_bel_dx.sweep_ron_min_vs_vbr(3200.0, 3300.0, 12, 1e-3, savefile=True, filename='aln_bel_dx_Ronsp_3300.csv')
    
    # out2 = GaN_cao.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='GaN_Cao_Ronsp_10000.csv')
    # out2 = SiC.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='SiC_Ronsp_10000.csv')
    # out2 = bGa2O3_x.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='bGa2O3_x_Ronsp_10000.csv')
    # out2 = bGa2O3_exp.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='bGa2O3_exp_Ronsp_10000.csv')
    # out2 = diamond300.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='diamond_300_Ronsp_10000.csv')
    # out2 = diamond600.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='diamond_600_Ronsp_10000.csv')
    # out2 = cbn.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='cbn_Ronsp_10000.csv')
    # out2 = aln_bel.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='aln_bel_Ronsp_10000.csv')
    # out2 = aln_bel_dx.sweep_ron_min_vs_vbr(9900.0, 10000.0, 12, 1e-3, savefile=True, filename='aln_bel_dx_Ronsp_10000.csv')
    
    #fig1 = plt.figure(dpi=200,figsize=[4.8,4.8])
    #ax1 = plt.subplot(111)
    #ax1.loglog(out[0],out[1])
    #ax2 = ax1.twinx()
    #ax2.loglog(out2[0],out2[1],color='r')
    #ax1.set_xlim([15.6,16.2])
    #ax1.set_ylim([3e-7,5e-7])
    #ax2.set_ylim([3e-4,4e-4])
    
    
    #test.find_minimum_ptotal(1500, 1e-3, 1.0)
    #__get_initial_guess__(1500, 1e-3)
    
    