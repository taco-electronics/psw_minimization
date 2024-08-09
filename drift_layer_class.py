# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:30:44 2024

@author: mapor
"""

import numpy as np
# import os
# import sys
# import matplotlib.pyplot as plt
# import multiprocessing as mp
# from functools import partial
# import time
# import matplotlib.transforms as transforms
# from matplotlib import rc, rcParams
import scipy.optimize as opt

import scipy.integrate as integrate

#import platform

#import originpro as op

class univ_consts:
    def __init__(self):
        self.eps0 = 8.85e-14
        self.q = 1.6e-19
        self.k = 8.617e-5
        self.h_c = 1.24e4 #cm-eV
        self.hbar_c = self.h_c/(2*np.pi)
        self.me_c2 = 0.511e6 #electron mass energy

class material(univ_consts):
    def __init__(self,eps=0,eg=0,nc=0,nv=0,ea=0,ii_model='',
                 ii_params_n=[],ii_params_p=[],
                 mun_model=None):
        super(material, self).__init__()
        self.eps = eps*self.eps0
        self.eg = eg
        self.nc = nc
        self.nv = nv
        self.ea = ea
        self.ii_model = ii_model
        self.ii_params_n = ii_params_n
        self.ii_params_p = ii_params_p
        #Function handle of one variable (Nd) must be provided for mun_model
        self.mun_model=mun_model
    
    def set_mat_params(self,eps,eg):
        self.eps = eps
        self.eg = eg
    
    def set_ii_params(self,ii_model,ii_params_n,ii_params_p):
        self.ii_model = ii_model
        self.ii_params_n = ii_params_n
        self.ii_params_p = ii_params_p
    
    def set_mobility_model(self,mun_model):
        self.mun_model = mun_model
    
    def alpha_n(self,E):
        #options: thornber, chynoweth, okuto_crowell, nth_poly
        #providing a model name outside of these options will set alpha_n=0
        if self.ii_model == 'thornber':
            return self.__alpha_thornber__(E, self.ii_params_n)
        elif self.ii_model == 'chynoweth':
            return self.__alpha_chynoweth__(E, self.ii_params_n)
        elif self.ii_model == 'okuto_crowell':
            return self.__alpha_okuto_crowell__(E,self.ii_params_n)
        elif self.ii_model == 'nth_poly':
            return self.__alpha_Nth_poly__(E, self.ii_params_n)
        else:
            return 0
    
    def alpha_p(self,E):
        #options: thornber, chynoweth, okuto_crowell, nth_poly
        #providing a model name outside of these options will set alpha_n=0
        if self.ii_model == 'thornber':
            return self.__alpha_thornber__(E, self.ii_params_p)
        elif self.ii_model == 'chynoweth':
            return self.__alpha_chynoweth__(E, self.ii_params_p)
        elif self.ii_model == 'okuto_crowell':
            return self.__alpha_okuto_crowell__(E,self.ii_params_p)
        elif self.ii_model == 'nth_poly':
            return self.__alpha_Nth_poly__(E, self.ii_params_p)
        else:
            return 0
    
    def __alpha_thornber__(self,E,params):
        Eg = params[0]
        mfp = params[1]
        Eith = params[2]
        Eph = params[3]
        Ekt = params[4]
        if E > 0:
            prefac = E/Eith
            num = 1.5*Eg
            if Eph > 0:
                den = E*mfp+E*E*mfp*mfp/Eph+Ekt
            else:
                den = E*mfp+Ekt
            return prefac*np.exp(-1*num/den)
        else:
            return 0
    
    def __alpha_chynoweth__(self,E,params):
        if E > 0:
            return params[0]*np.exp(-1*params[1]/E)
        else:
            return 0
    
    def __alpha_okuto_crowell__(self,E,params):
        if E > 0:
            return params[0]*np.power(E,params[1])*np.exp(-1*np.power(params[2]/E,params[3]))
        else:
            return 0
    
    def __alpha_Nth_poly__(self,E,params):
        if E > 0:
            exparg = 0
            for i in range(len(params)):
                exparg += params[-1*(i+1)]/np.power(E,i)
            return np.exp(exparg)
        else:
            return 0

class drift_layer(material):
    def __init__(self,material_params,drift_type='',
                 drift_doping_params=[]):
        #All material parameters must be given to initialize the class. I
        #may consider defaulting to the Si parameters in the future.
        self.temp=material_params['temp']
        eps=material_params['eps']
        eg=material_params['eg']
        nc=material_params['nc']*(self.temp/300)**1.5
        nv=material_params['nv']*(self.temp/300)**1.5
        ii_model=material_params['ii_model']
        #Pre-calculate temperature dependence of ii_params before passing to drift_layer class init
        ii_params_n=material_params['ii_params_n']
        ii_params_p=material_params['ii_params_p']
        mun_model=lambda n: material_params['mun_model'](n,self.temp)
        ea=material_params['ea']
        super(drift_layer,self).__init__(eps,eg,nc,nv,ea,ii_model,
                     ii_params_n,ii_params_p,mun_model)
        #Acceptable flags for self.drift_type: 
        #'const' - drift layer with constant doping Nd and thickness t
        #'var' - drift layer with variable doping. Actual thickness is
        #        assumed to be infinite; effective thickness is determined by the 
        #        breakdown voltage. Nd must be always > 0 (no multi-junction solutions permitted) 
        self.drift_type = drift_type
        #For 'const:
        #       self.dop_params[0] - doping
        #       self.dop_params[1] - thickness
        #For 'var':
        #       self.dop_params[0] - function reference for Nd(x)
        self.dop_params = drift_doping_params
        self.vbr = 0
        self.emax = 0
        self.emin = 0
        self.wbr = 0
    
    def __solve_w__(self,V):
        if self.drift_type == 'const':
            ni = np.sqrt(self.nc*self.nv)*np.exp(-1*self.eg/(2*self.k*self.temp))
            Vbi = self.eg/2+self.k*self.temp*np.log(self.dop_params[0]/ni)
            #print(Vbi)
            #print("%.2e"%V)
            return np.sqrt(2.0*self.eps*(Vbi+V)/(self.q*self.dop_params[0]))
        elif self.drift_type == 'var':
            #t0 = time.time()
            integrand = lambda y: y*self.dop_params(y)
            intout = lambda x: integrate.quad(integrand,0,x)[0]
            lhs = lambda x: V-(self.q/self.eps)*intout(x)
            root = opt.root_scalar(lhs,method='brentq', bracket=[0,0.1])
            #print('wtime:%.4f'%(time.time()-t0))
            return root.root
    
    def __w_emax_emin__(self,V):
        W = self.__solve_w__(V)
        #print(W)
        if self.drift_type == 'const':
            if W < self.dop_params[1]:
                emax = self.eprof_x(0,V,W)
                emin = 0
                return emax,emin,W
            else:
                emax = self.eprof_x(0,V,W)
                emin = self.eprof_x(self.dop_params[1],V,W)
                return emax,emin,self.dop_params[1]
        elif self.drift_type == 'var':
            int1 = integrate.quad(self.dop_params,0,W)
            emax = (self.q/self.eps)*int1[0]
            emin = 0
            return emax,emin,W
            
    def eprof_x(self,x,V,W):
        if self.drift_type == 'const':
            ni = np.sqrt(self.nc*self.nv)*np.exp(-1*self.eg/(2*self.k*self.temp))
            Vbi = self.eg/2+self.k*self.temp*np.log(self.dop_params[0]/ni)
            #W = self.__solve_w__(V)
            
            if W < self.dop_params[1]:
                emax=self.q*self.dop_params[0]*W/(self.eps)
            else:
                emax=(V+Vbi)/self.dop_params[1]+(self.q*self.dop_params[0]*self.dop_params[1])/(2*self.eps)
            mag = emax-self.q*self.dop_params[0]*x/(self.eps)
            if mag >= 0:
                return mag
            else:
                return 0
        elif self.drift_type == 'var':
            #W = self.__solve_w__(V)
            int1 = integrate.quad(self.dop_params,x,W)
            #int2 = integrate.quad(self.dop_params,0,x)
            if x <= W:
                return (self.q/self.eps)*int1[0]
            #(int1[0]-int2[0])
            else:
                return 0

    def __II_integrand__(self,x,V,W):
        if self.drift_type == 'const':
            #ni = np.sqrt(self.nc*self.nv)*np.exp(-1*self.eg/(2*self.k*300))
            #Vbi = self.eg/2+self.k*300*np.log(self.dop_params[0]/ni)
            ex = self.eprof_x(x,V,W)
        elif self.drift_type == 'var':
            ex = self.eprof_x(x,V,W)
            
        diff = lambda x: self.alpha_n(self.eprof_x(x,V,W))-self.alpha_p(self.eprof_x(x,V,W))
        exp_arg = integrate.quad(diff,0,x,epsabs=1e-12)    
        return self.alpha_n(ex)*np.exp(-1*exp_arg[0])

    def quad_Melec(self,V):
        W = self.__solve_w__(V)
        integ = lambda x: self.__II_integrand__(x,V)
        if self.drift_type == 'const':
            if W < self.dop_params[1]:
                ion_int = integrate.quad(integ,0,W,epsabs=1e-12)
                return 1/(1-ion_int[0])
            else:
                ion_int = integrate.quad(integ,0,self.dop_params[1],epsabs=1e-12)
                return 1/(1-ion_int[0])
        elif self.drift_type == 'var':
            ion_int = integrate.quad(integ,0,W,epsabs=1e-12)
            return 1/(1-ion_int[0])
    
    def quad_find_vbr(self,vinit):
        Vbr = vinit
        Vstep = 1
        Me = 0
        stepcount = 1
        while Me < 1e6:
           Me = self.quad_Melec(Vbr)
           print(Me)
           if Me < 0 and stepcount == 1:
               return Vbr
           elif Me < 0 and stepcount != 1:
               Vbr = Vbr/(1+Vstep)
               Vstep = 0.5*Vstep
               Vbr = Vbr*(1+Vstep)
               stepcount = stepcount + 1
           else:
               Vbr = Vbr*(1+Vstep)
               stepcount = stepcount + 1
        emax,emin,w = self.__w_emax_emin__(Vbr)
        return (Vbr,emax,emin,w)
    
    def quad_ionint(self,V):
        W = self.__solve_w__(V)
        integ = lambda x: self.__II_integrand__(x,V,W)
        if self.drift_type == 'const':
            if W < self.dop_params[1]:
                return integrate.quad(integ,0,W,epsabs=1e-12)[0]
            else:
                return integrate.quad(integ,0,self.dop_params[1],epsabs=1e-12)[0]
        elif self.drift_type == 'var':
            return integrate.quad(integ,0,W,epsabs=1e-12)[0]
   
    def quad_ionint_w_deriv(self,V):
       W = self.__solve_w__(V)
       integ = lambda x: self.__II_integrand__(x,V,W)
       if self.drift_type == 'const':
           if W < self.dop_params[1]:
               return integrate.quad(integ,0,W,epsabs=1e-12)[0]
           else:
               return integrate.quad(integ,0,self.dop_params[1],epsabs=1e-12)[0]
       elif self.drift_type == 'var':
           return integrate.quad(integ,0,W,epsabs=1e-12)[0] 
   
    def quad_find_vbr_ionint(self,vinit):
        Vbr = 1 
        #vinit
        stepcount = 1
        
        ionintlast = 0
        
        tol = 1
        
        swtype = (self.drift_type == 'var' or (self.drift_type == 'const' and self.dop_params[1] > 5e14))
        
        if swtype:
            Vstep = 1
        else:
            Vstep = 0.5
        multlast = 1+Vstep
        
        maxvbr = 1e16
        
        while tol > 1e-6 and Vbr < maxvbr:
           ionint = self.quad_ionint(Vbr)
           #print(ionint)
           #print(Vbr,self.dop_params)
           tol = np.abs(ionint-1)
           #print('iitime:%.4f'%(time.time()-t1))
           if ionint-1 > 0 and stepcount == 1:
               return Vbr
           elif ionint-1 > 0 and stepcount != 1:
               Vbr = Vbr/multlast
               Vstep = 0.5*Vstep
               Vbr = Vbr*(1+Vstep)
               multlast = 1+Vstep
               stepcount = stepcount + 1
           else:
                if np.abs(ionint-ionintlast) < 1e-6 and swtype:
                    Vbr = Vbr*(1+2*Vstep)
                    multlast = (1+2*Vstep)
                elif 1e-6 <= np.abs(ionint-ionintlast) and np.abs(ionint-ionintlast) < 1e-4 and swtype:
                    Vbr = Vbr*(1+1.25*Vstep)
                    multlast = (1+1.25*Vstep)
                else:
                    Vbr = Vbr*(1+Vstep)
                    multlast = (1+Vstep)
                ionintlast = ionint
                stepcount = stepcount + 1
        if Vbr < maxvbr:
            emax,emin,w = self.__w_emax_emin__(Vbr)
            #print(w)
            self.vbr = Vbr
            self.emax = emax
            self.emin = emin
            self.wbr = w
            return (Vbr,emax,emin,w)
        else:
            self.vbr = -1
            self.emax = -1
            self.emin = -1
            self.wbr = -1
            return (-1,-1,-1,-1)
    
    def bracket_find_vbr_ionint(self,vinit):
        vmin = 1
        vmax = 100 
        #vmin
        
        fmin = lambda x: self.quad_ionint(x)-1
        #while np.sign(fmin(vmin)) == np.sign(fmin(vmax)):
        #    vmax = 3.5*vmax
        
        #print(fmin(vmin))
        #print(fmin(vmax))
        
        conv = False
        while not conv:
            try:
                root = opt.root_scalar(fmin,method='brentq',bracket=[vmin,vmax])
                conv = True
            except:
                vmax = vmax*1.5
                #print(vmax)
                continue
        
        emax,emin,w = self.__w_emax_emin__(root.root)
        self.vbr = root.root
        self.emax = emax
        self.emin = emin
        self.wbr = w
        return (root.root,emax,emin,w)
    
    def newton_find_vbr_ionint(self,vinit):
        vinit = 1
        
        fmin = lambda x: self.quad_ionint(x)-1
        
        root = opt.root_scalar(fmin,method='secant',x0=vinit,x1=10000*vinit)
        
        w,emax,emin = self.__w_emax_emin__(root.root)
        self.vbr = root.root
        self.emax = emax
        self.emin = emin
        self.wbr = w
        return (root.root,emax,emin,w)
    
    def n_actual(self,nd):
        nc = self.nc*np.exp(-1*self.ea/(self.k*self.temp))
        t1 = np.sqrt(1+4*nd/nc)
        return 0.5*nc*(t1-1)
    
    def ron_sp(self,V):
        if self.drift_type == 'const':
            return self.dop_params[1]/(self.q*self.mun_model(self.dop_params[0])*self.n_actual(self.dop_params[0]))
            #w = self.__solve_w__(V)
            #if self.dop_params[1] < w:
                #return self.dop_params[1]/(self.q*self.mun_model(self.dop_params[0])*self.dop_params[0])
            #    return self.dop_params[1]/(self.q*self.mun_model(self.dop_params[0])*self.n_actual(self.dop_params[0]))
            #else:
            #    return w/(self.q*self.mun_model(self.dop_params[0])*self.n_actual(self.dop_params[0]))
        elif self.drift_type == 'var':
            w = self.__solve_w__(V)
            dron = lambda x: 1/(self.mun_model(self.dop_params(x))*self.dop_params(x))
            return (1/self.q)*integrate.quad(dron,0,w)[0]
    
    def coss_sp(self,V):
        if self.drift_type == 'const':
            w = self.__solve_w__(V)
            if self.dop_params[1] < w:
                return self.eps/self.dop_params[1]
            else:
                return self.eps/w
        elif self.drift_type == 'var':
            w = self.__solve_w__(V)
            return self.eps/w
    
    def eoss_sp(self,V):
        integ = lambda x: x*self.coss_sp(x)
        return integrate.quad(integ,0,V)[0]
    
    def sw_fom1(self,V,alpha):
        return self.ron_sp(V)*self.eoss_sp(alpha*V)

    def pmin_area(self,V,I,f,alpha):
        return 2*I*np.sqrt(f)*np.sqrt(self.sw_fom1(V,alpha))

# if __name__=='__main__':
#     def mun_gan(nd):
#         nf = 2e17
#         #alpha=2.0
#         #beta=0.7
#         #gam=1.0
#         mu_max=1.0e3
#         mu_min=55.0
#         bi = (mu_min+mu_max*(nf/nd))/(mu_min-mu_max)
#         return mu_max*(bi/(1+bi))
    
#     GaN_chyn_n_props = [2.11e9,3.689e7]
#     GaN_chyn_p_props = [4.39e6,1.8e7]
    
#     GaNparams = {'eps': 8.9,
#                   'eg':3.43,
#                   'nc':2.24e18,
#                   'nv':2.51e19,
#                   'ii_model':'chynoweth',
#                   'ii_params_n':GaN_chyn_n_props,
#                   'ii_params_p':GaN_chyn_p_props,
#                   'mun_model':mun_gan,
#                   'temp':300,
#                   'ea':0.0259}
    
#     testV = 1900
#     uni_drift_params = [1e12,10000e-4]
# #     a = ((1e16-5e15)/10e-4)
# #     tau = 20e-4
# #     linvar_drift_params = lambda x: 4e17*(1-np.exp(-x/tau))+2.5e16 
# #         #5e18*x+5e15
# #         #1e16*(1-np.exp(-x/tau))+9.2e15 
# #     #((1e16-5e15)/10e-4)*x+5e15
    
# #     testfig = plt.figure(dpi=200,figsize=[4.8,4.8])
# #     testax = plt.subplot(111)
    
# #     iifig = plt.figure(dpi=200,figsize=[4.8,4.8])
# #     iiax = plt.subplot(111)
    
# #     testx = np.linspace(0,10e-4,1000)
    
#     const_drift_layer = drift_layer(GaNparams,drift_type='const',
#                                     drift_doping_params=uni_drift_params)
    
#     import time
#     t1 = time.time()
#     out = const_drift_layer.bracket_find_vbr_ionint(1)
#     print(time.time()-t1)
#     out2 = const_drift_layer.quad_find_vbr_ionint(1)
#     print(time.time()-t1)
    
#     var_drift_layer = drift_layer(GaNparams,drift_type='var',
#                                     drift_doping_params=linvar_drift_params)
    
#     eprof_const = np.zeros(1000)
#     eprof_var = np.zeros(1000)
    
#     ii_const = np.zeros(1000)
#     ii_var = np.zeros(1000)
    
#     wconst = const_drift_layer.__solve_w__(testV)
#     wvar = var_drift_layer.__solve_w__(testV)
    
#     for i,x in enumerate(testx):
#         eprof_const[i] = const_drift_layer.eprof_x(x, testV, wconst)
#         eprof_var[i] = var_drift_layer.eprof_x(x, testV, wvar)
#         ii_const[i] = const_drift_layer.__II_integrand__(x, testV, wconst)
#         ii_var[i] = var_drift_layer.__II_integrand__(x, testV, wvar)
    
#     testax.plot(testx,eprof_const)
#     testax.plot(testx,eprof_var)
    
#     iiax.plot(testx,ii_const)
#     iiax.plot(testx,ii_var)
    
#     #test1 = const_drift_layer.quad_find_vbr(1000)
#     #test2 = var_drift_layer.quad_find_vbr(0.8*test1[0])
#     t1 = time.time()
#     test1 = const_drift_layer.quad_find_vbr_ionint(1000)
#     t2 = time.time() 
#     print('const runtime, loop:%.4f'%(t2-t1))
#     test2 = var_drift_layer.quad_find_vbr_ionint(0.5*test1[0])
#     t3 = time.time()
#     print("var runtime, loop:%.4f"%(t3-t2))
#     test3 = var_drift_layer.bracket_find_vbr_ionint(0.5*test1[0])
#     t4 = time.time()
#     print("var runtime, brentq w/man bounds:%.4f"%(t4-t3))
#     # test3 = var_drift_layer.newton_find_vbr_ionint(0.5*test1[0])
#     # t5 = time.time()
#     # print("var runtime, secant:%.4f"%(t4-t4))

#     ii_plot = plt.figure(dpi=200,figsize=[4.8,4.8])
#     ii_ax = plt.subplot(111)
    
#     eoss_plot = plt.figure(dpi=200,figsize=[4.8,4.8])
#     eoss_ax = plt.subplot(111)
    
#     ronsp_plot = plt.figure(dpi=200,figsize=[4.8,4.8])
#     ronsp_ax = plt.subplot(111)
    
#     vtest = np.logspace(1,4,int(1e2))
#     ii_test = np.zeros(int(1e2))
#     ii_test_const = np.zeros(int(1e2))
#     vcoss = np.linspace(1,1900,int(1e2))
#     coss_test = np.zeros(int(1e2))
#     coss_test_const = np.zeros(int(1e2))
#     rsp_test = np.zeros(100)
#     rsp_test_const = np.zeros(100)
    
#     for i, v in enumerate(vtest):
#         ii_test[i] = var_drift_layer.quad_ionint(vcoss[i])
#         ii_test_const[i] = const_drift_layer.quad_ionint(vcoss[i])
#         coss_test[i] = var_drift_layer.eoss_sp(vcoss[i])
#         coss_test_const[i] = const_drift_layer.eoss_sp(vcoss[i])
#         rsp_test[i] = var_drift_layer.ron_sp(vcoss[i])
#         rsp_test_const[i] = const_drift_layer.ron_sp(vcoss[i])
    
#     ii_ax.loglog(vcoss,ii_test)
#     ii_ax.loglog(vcoss,ii_test_const)
#     eoss_ax.plot(vcoss,coss_test)
#     eoss_ax.plot(vcoss,coss_test_const)
#     ronsp_ax.plot(vcoss,rsp_test,vcoss,rsp_test_const)
#     print(var_drift_layer.eoss_sp(test3[0]))
#     print(var_drift_layer.ron_sp(test3[0]))
    #.semilogx(vtest,ii_test)

# def eprof_x(x,V,t,Nd,Eg,epsr):
#     Vbi = Eg/2+0.0259*np.log(Nd/1e-10)
#     W = np.sqrt((2*epsr*eps0/(q*Nd))*(Vbi+V))
#     if W < t:
#         emax=q*Nd*W/(epsr*eps0)
#     else:
#         emax=(V+Vbi)/t+(q*Nd*t)/(2*eps0*epsr)
#     mag = emax-q*Nd*x/(eps0*epsr)
#     if mag >= 0:
#         return mag
#     else:
#         return 0

# def integrand(x,V,t,Nd,ii_model,matprops,ii_prms_n,ii_prms_p):
#     Eg = matprops[0]
#     epsr = matprops[1]
    
#     ex = eprof_x(x,V,t,Nd,Eg,epsr)    
#     diff = lambda x: ii_model(eprof_x(x,V,t,Nd,Eg,epsr),ii_prms_p)-ii_model(eprof_x(x,V,t,Nd,Eg,epsr),ii_prms_n)
#     exp_arg = integrate.quad(diff,0,x,epsabs=1e-12)    
#     return ii_model(ex,ii_prms_p)*np.exp(-1*exp_arg[0])

# def quad_Melec(V,t,Nd,ii_model,matprops,ii_prms_n,ii_prms_p):
#     Eg = matprops[0]
#     epsr = matprops[1]
#     [emax,emin] = find_emax_min(V,t,Nd,Eg,epsr)
#     Vbi = Eg/2+0.0259*np.log(Nd/1e-10)
#     W = np.sqrt((2*epsr*eps0/(q*Nd))*(Vbi+V))
#     integ = lambda x: integrand(x,V,t,Nd,ii_model,matprops,ii_prms_n,ii_prms_p)
#     if W < t:
#         ion_int = integrate.quad(integ,0,W,epsabs=1e-12)
#     else:
#         ion_int = integrate.quad(integ,0,t,epsabs=1e-12)
#     if ion_int[0] == float(1):
#         return [np.inf,emax,emin,t]
#     else:
#         if W < t:
#             return [1/(1-ion_int[0]),emax,emin,W]
#         else:
#             return [1/(1-ion_int[0]),emax,emin,t]

# def quad_find_Vbr(ii_model,matprops,ii_prms_n,ii_prms_p,verbose,t_Nd):
#     Vbr = 1
#     # if matprops[0] < 2:
#     #     Vstep = 5
#     # else:
#     #     Vstep = 1*(t/10e-4)
#     Vstep = 0.5
#     Me = 0
#     stepcount = 1
#     while Me < 1e6:
#        [Me,emax,emin,w] = quad_Melec(Vbr,t_Nd[0],t_Nd[1],ii_model,matprops,ii_prms_n,ii_prms_p)
#        if Me < 0 and stepcount == 1:
#            return Vbr
#        #print(Me)
#        elif Me < 0 and stepcount != 1:
#            Vbr = Vbr/(1+Vstep)
#            Vstep = 0.5*Vstep
#            stepcount = stepcount + 1
#        else:
#            Vbr = Vbr*(1+Vstep)
#            stepcount = stepcount + 1
#        if verbose:
#            print("%.8f,%.8f"%(Vbr,Me))
#     return (Vbr,emax,emin,w)