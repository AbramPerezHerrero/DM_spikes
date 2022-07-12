# -*- coding: utf-8 -*-
"""
Code to obtain the properties of the dark matter spikes.

@author: aph278
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from functools import partial
from scipy import optimize
from scipy.integrate import quad
from scipy import interpolate
import time
from tqdm import tqdm
import datetime
import matplotlib as mpl
import multiprocessing as mul
import Functions as ds
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Plot options
mpl.rcParams.update({'font.size': 16,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['legend.edgecolor'] = 'inherit'

class dm_spike(object):
    '''
    Class where three different initial models of dark matter density profiles
    and their corresponding dark matter spike can be established.
    We have considered that the dark matter spike is created after 
    the adiabatic growth of a black hole in its interior.  
    
    The models considered are:
        * Hernquist
        * Power-law
        * Super-Navarro-Frenk-White (sNFW)
        
        
    Attributes
    ----------
    M_halo : FLOAT
     Total mass of the initial halo. Must be in M_sun units.
     
    M_bh : FLOAT
     Black hole mass.Must be in M_sun units.
     
    Scale_param :  FLOAT
     Scale parameter of the Hernquist model.
     
    n_step_z : INT
     Number of steps to obtain the integral over z.
     
    n_step_u : INT
     Number of steps to obtain the integral over u.
     
    Initial_profile : STRING
     Name of the initial density model considered (Hernquist,sNFW,Power-law)
     
    gamma : FLOAT, optional
     Power index considered in the Power-law model. The default is 1.

     
    Methods
    -------
    initial_density(r)
        * Compute the initial density profile for a given radius. 
    plot_initial_density(scale_mode)
        * Plot the initial density profile for a given array of radius.
    spike_density_radius(r)
        * Compute the dark matter spike density for a given radius and a given model.
    spike_density_profile(r,multiprocess=True,nproces=mul.cpu_count())
        * Compute the dark matter spike for a set of radius and a given model.
    read_spike_density( pathfile,namefile)
        * Read and save a txt with the DM spike density obtained.
    plot_spike_density(scale_mode)
        * Plot the DM spike density.
        
    '''
    def __init__(self,M_halo,M_bh,Scale_param,n_step_z,n_step_u,Initial_profile,gamma=1):
          '''
         Initialise class variables.

        Parameters
        ----------
        M_halo : FLOAT
            Total mass of the initial halo.
            Must be in M_sun units.
        M_bh : FLOAT
            Black hole mass.
            Must be in M_sun units.
        Scale_param :  FLOAT
            Scale parameter of the Hernquist model.
        n_step_z : INT
            Number of steps to obtain the integral over z.
        n_step_u : INT
            Number of steps to obtain the integral over u.
        Initial_profile : STRING
            Name of the initial density model considered (Hernquist,sNFW,Power-law)
        gamma : FLOAT, optional
            Power index considered in the Power-law model. The default is 1.

        Returns
        -------
        None.

          '''
           # Change the units to Mass [GeV] and Distance [cm]
        
          self.M_halo = ((M_halo*u.M_sun).to('kg')*const.c**2).to('GeV')
          self.Scale_param = (Scale_param*u.kpc).to('cm')
          self.M_bh = ((M_bh*u.M_sun).to('kg')*const.c**2).to('GeV')
          self.n_step_z = n_step_z
          self.n_step_u = n_step_u
          self.Initial_profile = Initial_profile
          self.G = (const.G/const.c**4).to('cm/GeV')
          self.Rs = 2*self.G*self.M_bh
          self.m = self.M_bh/self.M_halo
          self.p0 = self.M_halo/(2*np.pi*self.Scale_param**3)
          self.spike_density = 0.
          self.gamma = gamma
          self.alpha_gamma = 0.68
    def plot_initial_density(self,r,scale_mode):
        '''
        Method to plot the initial density profile considered.

        Parameters
        ----------
        r : ARRAY
            Radii array in which the density is evaluated.
            Must be in Rs units.
        
        scale_mode : STRING
            'log' to put in logarithm scale. Any string for normal scale.
            
        Returns
        -------
        PLot.
        '''
        if self.Initial_profile == 'Hernquist'  :                
            if scale_mode == 'log':
                 plt.plot(r*self.Rs.to('pc').value,ds.density_Hernquist(r*self.Rs.to('cm').value
                          ,self.M_halo.value,self.Scale_param.to('cm').value)
                          ,'m-',markersize=5,label=r'Hernquist')
                 plt.xscale('log')
                 plt.yscale('log')
                 plt.legend()
                 plt.xlabel(r'$r [pc]$')
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')
            else:
                 plt.plot(r*self.Rs.to('pc').value,ds.density_Hernquist(r*self.Rs.to('cm').value
                          ,self.M_halo.value,self.Scale_param.to('cm').value)
                          ,'m-',markersize=5,label=r'Hernquist')
                 plt.xlabel(r'$r [pc]$')
                 plt.legend()
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')  
                 
        if self.Initial_profile == 'Power-law'  :                
            if scale_mode == 'log':
                 plt.plot(r*self.Rs.to('pc').value,ds.density_powerlaw(r*self.Rs.to('cm').value
                          ,self.M_halo.value,self.Scale_param.to('cm').value,self.gamma)
                          ,color='orange',markersize=5,label=r'Power-law')
                 plt.xscale('log')
                 plt.yscale('log')
                 plt.legend()
                 plt.xlabel(r'$r [pc]$')
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')
            else:
                 plt.plot(r*self.Rs.to('pc').value,ds.density_powerlaw(r*self.Rs.to('cm').value
                           ,self.M_halo.value,self.Scale_param.to('cm').value)
                          ,color='orange',markersize=5,label=r'Power-law')
                 plt.xlabel(r'$r [pc]$')
                 plt.legend()
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')   
                 
        if self.Initial_profile == 'sNFW'  :             
            if scale_mode == 'log':
                 plt.plot(r*self.Rs.to('pc').value,ds.density_sNFW(r*self.Rs.to('cm').value
                          ,self.M_halo.value,self.Scale_param.to('cm').value)
                          ,'c-',markersize=5,label=r'sNFW')
                
                 plt.xscale('log')
                 plt.yscale('log')
                 plt.legend()
                 plt.xlabel(r'$r [pc]$')
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')
            else:
                 plt.plot(r*self.Rs.to('pc').value,ds.density_sNFW(r*self.Rs.to('cm').value
                          ,self.M_halo.value,self.Scale_param.to('cm').value)
                          ,'c-',markersize=5,label=r'sNFW')
                 plt.xlabel(r'$r [pc]$')
                 plt.legend()
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')        
                 
    def initial_density(self,r):
        '''
        Method to compute the intial density profile for a given radius.

        Parameters
        ----------
        r : FLOAT
            Radius in which the density is evaluated.
            Must be in Rs units.

        Returns
        -------
        FLOAT
            Initial density profile evaluate in radius r.

        '''
        if self.Initial_profile == 'Hernquist'  :  
                return ds.density_Hernquist(r*self.Rs.to('cm').value,self.M_halo.value,self.Scale_param.to('cm').value)
        if self.Initial_profile == 'sNFW'  :  
                return ds.density_sNFW(r*self.Rs.to('cm').value,self.M_halo.value,self.Scale_param.to('cm').value)
        if self.Initial_profile == 'Power-law'  :  
                return ds.density_powerlaw(r*self.Rs.to('cm').value,self.M_halo.value,self.Scale_param.to('cm').value,self.gamma)
            
    def spike_density_radius(self,r):
        '''
        Method to compute the dark matter spike density for a given radius.

        Parameters
        ----------
        r : FLOAT
            Radius in which the density is evaluated.
            Must be in Rs units.


        Returns
        -------
        FLoat
            DM spike density in (GeV/cm3) units at radius 'r'.

        '''
        if self.Initial_profile == 'Hernquist'  :  
                return ds.density_DMspike_Hernquist(r,self.M_halo.value,self.M_bh.value
                                                ,self.Scale_param.value,self.n_step_z)
        if self.Initial_profile == 'sNFW'  :  
                return ds.density_DMspike_sNFW(r,self.M_halo.value,self.M_bh.value
                                     ,self.Scale_param.value,self.n_step_z,self.n_step_u)
        if self.Initial_profile == 'Power-law'  :  
                return ds.density_DMspike_powerlaw(r,self.M_halo.value
                        ,self.M_bh.value,self.Scale_param.value,self.gamma,self.alpha_gamma) 
            
    def spike_annihilation(self,m_x,cross_section,t_bh):
        '''
        Method to change the dark matter spike to a dark matter spike considering
        the dark matter self-annihilation

        Parameters
        ----------
        m_x : FLOAT
            Mass of the dark matter particle
            Must be in GeV units.
        cross-section : FLOAT
            cross-section multiply by the velocity of the dark matter particles.
            Must be in cm^3/s units.

        Returns
        -------
        None

        '''        
        dens_core = m_x/(cross_section*t_bh)
        self.dens_core =dens_core
        inter = interpolate.interp1d(self.radii_vector, self.spike_density)
        solve = lambda r: inter(r)-dens_core
        r_lim = optimize.brentq(solve,5,np.max(self.radii_vector))
        self.r_lim =  r_lim
        self.spike_density = np.concatenate([np.ones(len(self.spike_density[self.radii_vector<=r_lim])
                                             )*dens_core,self.spike_density[self.radii_vector>r_lim]])
        
        
    def spike_density_profile(self,r,multiprocess=False,nproces=mul.cpu_count()):
        '''
        Method to compute the DM spike density profile for a given set of radii.

        Parameters
        ----------
        r : FLOAT
            Radius in which the density is evaluated.
            Must be in Rs units and greater than 4Rs.
        multiprocess : BOOLEAN, optional
            To activate the multiprocessing mode, this mode can be useful to
            reduce the running time but can give some problems.
            . The default is False.
        nproces : INT, optional
            If you activate multiprocessing mode, 
            you can select the number of processors you want to use.
            . The default is mul.cpu_count().
        
        This method prints the estimated time it will take to run. 
                 
        The multiprocessing is useful but it depends on your S.O. 
        
        Returns
        -------
        list
            First elemenent of the list: DM spike density for the given radii 
            in (GeV/cm3) units.
            Second elemenent of the list: Radii selected to evaluate the 
            DM spike density.

        '''
        self.radii_vector = r
        self.break_radius()
        r = self.radii_vector[self.radii_vector<=self.r_h*0.2]
        r2 =  self.radii_vector[self.radii_vector>self.r_h*0.2]
        if multiprocess and not self.Initial_profile == 'Power-law' :
            pool = mul.Pool(nproces)
            if self.Initial_profile == 'Hernquist'  :  
                part = partial(ds.density_DMspike_Hernquist,M_halo=self.M_halo.value,
                             M_bh=self.M_bh.value,a=self.Scale_param.value
                             ,n_step_z=self.n_step_z)
                density = np.asarray(pool.map(part,r))

                
            if self.Initial_profile == 'sNFW'  :  
                part = partial(ds.density_DMspike_sNFW,M_halo=self.M_halo.value,
                             M_bh=self.M_bh.value,a=self.Scale_param.value
                             ,n_step_z=self.n_step_z,n_step_u=self.n_step_u)
                density = np.asarray(pool.map(part,r))

        else:
            i = 1
            if self.Initial_profile == 'Hernquist'  :  
                density = np.empty((len(r)),dtype=float)
                start = time.time()
                density[0]=ds.density_DMspike_Hernquist(r[0],self.M_halo.value,self.M_bh.value
                                                       ,self.Scale_param.value,self.n_step_z)
                end = time.time()
                t = end -start
                print('Estimated time it takes :'+ str(np.round(t*len(r)/60,2))+ ' min' )
                
                for r1 in tqdm(r[1:len(r)]):
                    density[i] = ds.density_DMspike_Hernquist(r1,self.M_halo.value,self.M_bh.value
                                                ,self.Scale_param.value,self.n_step_z)
                    i += 1


            
            if self.Initial_profile == 'sNFW'  :  
                density = np.empty((len(r)),dtype=float)
                start = time.time()
                density[0]=ds.density_DMspike_sNFW(r[0],self.M_halo.value,self.M_bh.value
                                     ,self.Scale_param.value,self.n_step_z,self.n_step_u)
                end = time.time()
                t = end -start
                print('Estimated time it takes :'+ str(np.round(t*len(r)/60,2))+ ' min' )
                for r1 in tqdm(r[1:len(r)]):
                        
                    density[i] = ds.density_DMspike_sNFW(r1,self.M_halo.value,self.M_bh.value
                                     ,self.Scale_param.value,self.n_step_z,self.n_step_u)
                    i += 1

            
            if self.Initial_profile == 'Power-law'  :  
                density = np.empty((len(r)),dtype=float)
                start = time.time()
                density[0] = ds.density_DMspike_powerlaw(r[0],self.M_halo.value
                        ,self.M_bh.value,self.Scale_param.value,self.gamma,self.alpha_gamma) 
                end = time.time()
                t = end -start
                print('Estimated time it takes :'+ str(np.round(t*len(r)/60,2))+ ' min' )
                
                for r1 in tqdm(r[1:len(r)]):
                    density[i] = ds.density_DMspike_powerlaw(r1,self.M_halo.value
                        ,self.M_bh.value,self.Scale_param.value,self.gamma,self.alpha_gamma) 
                    i += 1
        if len(r2) > 0 :
            density_2 = self.initial_density(r2)
            self.spike_density = np.concatenate((density,density_2))
        else:
            self.spike_density = density
        return [self.spike_density,self.radii_vector]
    def read_spike_density(self, pathfile,namefile):
         '''
         Method to read a DM spike density profile stored
          in the txt format implemented in this code. 
          
        Parameters
        ----------
        pathfile :  STRING
            Path where the file is located
        namefile : STRING
            File name.

        Returns
        -------
        list
            First elemenent of the list: DM spike density for the given radii 
            in (GeV/cm3) units.
            Second elemenent of the list: Radii selected to evaluate the 
            DM spike density.

         '''
         pathfile = pathfile+namefile
         file = np.loadtxt(pathfile,comments='#')
         density = file[:,0]
         r = file[:,1]
         self.M_halo=((10**float(namefile.split('_')[4])*u.M_sun).to('kg')*const.c**2).to('GeV')
         self.M_bh=((10**float(namefile.split('_')[2])*u.M_sun).to('kg')*const.c**2).to('GeV')
         self.Rs=(2*(10**float(namefile.split('_')[2])*u.M_sun).to('kg')*const.G/const.c**2).to('cm')
         self.Scale_param=(float(namefile.split('_')[3])*u.kpc).to('cm')
         self.Initial_profile = namefile.split('_')[1]
         self.spike_density = density
         self.radii_vector = r
         return [density,r]
    def save_txt_density(self,pathfile):
        '''
         Method to save the dark matter density in a txt file, 
         with the appropriate format for reading with this code.
         
        Parameters
        ----------
        pathfile :  STRING
            Path where the file will locate

        Returns
        -------
        None.

        '''
        if any(self.spike_density > 0):
                 M_bh = (self.M_bh/const.c**2).to('M_sun')
                 M_halo = (self.M_halo/const.c**2).to('M_sun')
                 line1 = 'Dark matter spike values considering an initial  : '+self.Initial_profile+': density (GeV/cm^-3), radius (Rs)'
                 line2='MassBH='+str(np.round(M_bh.value,2))+' (M_sun)'+' ,a='+str(self.Scale_param.to('kpc').value)+' (kpc)'+' ,Mh='+str(np.round(M_halo.value,2))+' (M_sun)'
                 today = datetime. datetime. now()
                 date_time = today. strftime("%m/%d/%Y, %H:%M:%S")
                 line3 = date_time
                 namefile = 'Density_'+self.Initial_profile+'_'+str(np.log10(M_bh.value))+'_'+str(
                           self.Scale_param.to('kpc').value)+'_'+str(np.log10(M_halo.value))+'_'
                 data = np.column_stack([self.spike_density, self.radii_vector])
                 datafile_path = pathfile + namefile+'.txt'
                 np.savetxt(datafile_path , data,header='\n'.join([line1, line2,line3]),
                        fmt='%10.5f')
   
        else:
             
             return None
         
    def plot_spike_density(self,scale_mode):
        '''
        Method to plot the DM spike density profile as a funciton of radii.

        Parameters
        ----------
        scale_mode : STRING
            If it is 'log' the plot will be on a logarithmic scale, 
            otherwise on a normal scale..

        Returns
        -------
        None.

        '''
        r = self.radii_vector*(self.Rs.to('pc')).value
        if self.Initial_profile == 'Hernquist':
             color='mediumorchid'
        elif self.Initial_profile == 'sNFW':
             color='mediumturquoise'
        else:
             color='darkorange'
             
        if any(self.spike_density > 0) :                
             if scale_mode == 'log':
                 plt.plot(r,self.spike_density,color=color
                          ,markersize=5,label=str(self.Initial_profile))
                 plt.xscale('log')
                 plt.yscale('log')
                 plt.legend()
                 plt.xlabel(r'$r [pc]$')
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')
             else:
                 plt.plot(r,self.spike_density
                          ,color=color,markersize=5,label=str(self.Initial_profile))
                 plt.xlabel(r'$r [pc]$')
                 plt.legend()
                 plt.ylabel(r'$\rho$ [GeV cm$^{-3}$])')   
        else:
             return None
    def break_radius(self):
        '''
        Method to obtain the radius of influence of the black hole.
        It is needed to run first the method spike_density_profile or read_spike_density
        Parameters
        ----------
        None
        
        Returns
        -------
        None.

        '''       
        if self.Initial_profile == 'sNFW'  : 
           r_h_function = lambda r1 : (3*self.M_halo.value/4)*(np.log(1+r1/self.Scale_param.to('cm').value
                                                     )-(r1/self.Scale_param.to('cm').value
                                                       )/(1+r1/self.Scale_param.to('cm').value))-2*self.M_bh.value
                                                          
           self.r_h = ( optimize.brentq(r_h_function,0,1e47))/self.Rs.to('cm').value 
           
        if self.Initial_profile == 'Hernquist'  : 
          
           r_h_function = lambda r1 : (self.M_halo.value*(r1/self.Scale_param.to('cm').value)**2/(1+r1/self.Scale_param.to('cm').value)**2)-2*self.M_bh.value 
           self.r_h = ( optimize.brentq(r_h_function,0,1e47))/self.Rs.to('cm').value
    
        if self.Initial_profile == 'Power-law'  : 
           
           r_h_function = lambda r1 : (self.M_halo.value*(r1/self.Scale_param.to('cm').value)**2/(1+r1/self.Scale_param.to('cm').value)**2)-2*self.M_bh.value 
           self.r_h = ( optimize.brentq(r_h_function,0,1e47))/self.Rs.to('cm').value       
    
    