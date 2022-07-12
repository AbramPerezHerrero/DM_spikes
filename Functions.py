# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:20:35 2022

@author: aph278
"""
import numpy as np
from scipy.integrate import quad,simpson
from scipy import interpolate
from scipy import optimize
from astropy import constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpmath as mp



def density_Hernquist(r,M_halo,a):
    '''
    Hernquist density profile of Dark matter.
    Recomendation: Use the same units for 'r'
    and the scale factor 'a'.
    
    Parameters
    ----------
    r : FLOAT
        Radius where density is evaluated.
    M_halo : FLOAT
        Total mass of the halo.
    a : FLOAT
        Scale parameter of the model.

    Returns
    -------
    FLOAT
        Density considering a Hernquist density profile
        value at radius 'r'.
    '''
    return M_halo*a/(2*np.pi*r*(a+r)**3)

def density_powerlaw(r,M_halo,a,gamma):
    '''
    Power-law density profile of Dark matter.
    Recomendation: Use the same units for 'r'
    and the scale factor 'a'.
    
    Parameters
    ----------
    r : FLOAT
        Radius where density is evaluated.
    M_halo : FLOAT
        Total mass of the halo.
    a : FLOAT
        Scale parameter of the model.
    gamma : FLOAT
        Power law parameter.
    Returns
    -------
    FLOAT
        Density considering a Power-law density profile
        value at radius 'r'.
    '''
    return (M_halo/(2*np.pi*a**3))*(r/a)**(-gamma) 
def density_sNFW(r,M_halo,a):
    '''
    Super-Navarro-Frenk_White density profile of Dark matter.
    Recomendation: Use the same units for 'r'
    and the scale factor 'a'.
    
    Parameters
    ----------
    r : FLOAT
        Radius where density is evaluated.
    M_halo : FLOAT
        Total mass of the halo.
    a : FLOAT
        Scale parameter of the model.

    Returns
    -------
    FLOAT
        Density considering a sNFW density profile
        value at radius 'r'.
    '''
    return 3*M_halo*np.sqrt(a)/(16*np.pi*r*(r+a)**(5/2))


def density_DMspike_powerlaw(r,M_halo,M_bh,a,gamma,alpha_gamma=0.62):
    '''
    Dark matter spike density considering an initial DM power law
    density profile.
    # Article: https://arxiv.org/abs/astro-ph/9906391

    Parameters
    ----------
    r : FLOAT
        Radius in which the density is evaluated.
        Must be in Rs units.
    M_halo : FLOAT
        Total mass of the initial halo.
        Must be in M_sun units.
    M_bh : FLOAT
        Black hole mass.
        Must be in M_sun units.
    a :  FLOAT
        Scale parameter of the Hernquist model.
    gamma : FLOAT
        Power law parameter.

    Returns
    -------
    FLOAT
    DM density value at radius 'r'.

    '''
    p_0=M_halo/(2*np.pi*a**3)
    # Approximate parameter of the equation (9)
    G = ((const.G/const.c**4).to('cm/GeV')).value
    Rs = 2*G*M_bh
    
    r=r*Rs
    
    R_sp=alpha_gamma*a*(M_bh/(p_0*a**3))**(1/(3-gamma))
    p_r=p_0*(R_sp/a)**(-gamma)
    g_alpha=(1-4*Rs/r)**3
    gamma_sp=(9-2*gamma)/(4-gamma)
    return p_r*g_alpha*(R_sp/r)**gamma_sp

def density_DMspike_Hernquist(r,M_halo,M_bh,a,n_step_z=6):
    '''
    Dark matter spike density considering an initial DM Hernquist 
    density profile.
    Parameters
    ----------
    r : FLOAT
        Radius in which the density is evaluated.
        Must be in Rs units.
    M_halo : FLOAT
        Total mass of the initial halo.
        Must be in M_sun units.
    M_bh : FLOAT
        Black hole mass.
        Must be in M_sun units.
    a :  FLOAT
        Scale parameter of the Hernquist model.
    n_step_z : INT
        Number of steps to obtain the integral over z.

    Returns
    -------
    FLOAT
    DM density value at radius 'r'.
    '''
    m = M_bh/M_halo
    G = ((const.G/const.c**4).to('cm/GeV')).value
    Rs = 2*G*M_bh
    
    def e_f_max(x):
        '''
        Maximum energy that a bound particle could have
        after the formation of the black hole considering
        the capture effect.
        
        Parameters
        ----------
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Maximum energy value.

        '''
        return (m/x)*(1-(8*m*G*M_halo/(x*a)))
    
    def L_max(e_f,x):
        '''
        Maximum angular momentum that a bound particle could have
        after the formation of the black hole.       

        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        x : FLOAT
            Dimensionless radius r/a.
            
        Returns
        -------
        FLOAT
            Maximum angular momentum value.

        '''
        return (2*x**2*(m/x-e_f))**(1/2)
    
    # Minimum angular momentum that a bound particle could have 
    # after the formation of the black hole considering the capture effect.
    L_min = 4*m*(G*M_halo/a)**(1/2)


    def limits(e_i,L):
        '''
        Calculates the limits of the radial action integrand considering
        the initial Hernquist density profile.

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        list
          List with the value of the two positive roots of the integrand.
          [x_-, x_+]

        '''
        # List with the coefficients of numerator integrand
        coeff = [-2*e_i,2*(1-e_i),-L**2,-L**2]
        # Obtain the roots
        roots = np.roots(coeff)
        # Choose only the positive roots
        roots = roots[roots>=0]
        roots.sort()
        # Check if there are not complex roots
        if any(np.iscomplex(roots)) or len(roots)<2:
            warnings.warn('Complex roots in  theradial action integrand ')
            return [0,0]
        
        # Check what of the possible three roots have values positive of 
        # the integrand between them.
        #x2 = np.random.uniform(roots[0],roots[1],5)
        #check_values = -2*e_i*x2**3+2*(1-e_i)*x2**2-L**2*x2-L**2
        
        if len(roots)>2:
            return [roots[1],roots[2]]
        else:
            return [roots[0],roots[1]]

         
    def Irh(e_i,L):   
      '''
        Radial invariant action for the Hernquist density potential in
        dimensionless variables.

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        FLOAT
        Th invariant radial action 
      '''
      # List with the coefficients of numerator integrand 
      coeff = [-2*e_i,2*(1-e_i),-L**2,-L**2]
     # Integran 
      def integrand(x1):
          '''
          Radial invariant action integrand in dimensionless variables.

          Parameters
          ----------
          x1 : FLOAT
                Radius.

         Returns
         -------
         FLOAT
           Radial invariant action integrand value.
        '''
          return (np.polyval(coeff,x1)/(x1**2*(x1+1)))**(1/2)
      # Obtain the limits of the radial action
      lim = limits(e_i,L)
      # Compute the integral.
      return quad(integrand,lim[0],lim[1],epsrel=1e-3)[0] 

    def IrB(e_f,L):
        '''
        Radial adiabatic invariant action for the point mass potential in
        dimensionless variables.

        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        FLOAT
           Radial invariant action for a point mass potential.
        '''
        return np.pi*(m/(2*e_f)**(1/2)-L)
    
    def e_f_to_e_i(e_f,L):
        '''
        Find the energy of the particles before the
        formation of the black hole e_i given the energy after 
        the formation of the black hole e_f.
        
        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        L : FLOAT
            Angular momentum.


        Returns
        -------
        FLOAT
            Initial energy of the particles.

        '''
        def radial_action_conservation(e_i):
            '''
            Radial invariant action conservation,i.e., before and after 
            the formation of the black hole.

            Parameters
            ----------
            e_i : FLOAT
                Energy before the formation of the black hole.

            Returns
            -------
            FLOAT
                The difference between the radial actions.

            '''
            dif=Irh(e_i[0],L)-IrB(e_f,L) if e_i[0] <= 1 else 1e10
            return dif
        # We minimize the difference to find the root.
        return optimize.fsolve(radial_action_conservation,0.1)[0]

    def f_Hq(e_i):
        '''
        The phase-space distribution function of the Hernquist model.
        # Article: https://arxiv.org/abs/1305.2619
        # https://ui.adsabs.harvard.edu/abs/2008gady.book.....B/

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.

        Returns
        -------
        FLOAT
            Phase-space distribution function evaluate 
            in energy point e_i.

        '''
        if e_i != 0 and e_i != 1:
            return e_i**(1/2)/(1-e_i)**2*((1-2*e_i)*(8*e_i**2-8*e_i-3)+
                  (3*np.arcsin(e_i**(1/2))/((e_i*(1-e_i))**(1/2))))
        else:
            return 0
        
    def integrand(u,z,x):
        '''
        Integrand of the Dark matter spike density considering
        an initial Hernnquist density profile.

        Parameters
        ----------
        u : FLOAT
            Parameter relate with the energy of the particles.
        z: FLOAT
            Parameter relate with the angular momentum of the particles.
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Integrand of DM spike value dudz.

        '''
        # Check that the parameter u is  different to zero  since
        # the zero energy is not considered.
        if u != 0:
            # Reset variable change
            e_f = u*e_f_max(x)
            L = (z*L_max(e_f,x)**2+(1-z)*L_min**2)**(1/2)
            ves =e_f_to_e_i(e_f,L)
            # Integrand value 
            return ((L_max(e_f,x)**2-L_min**2)/(1-z))**(1/2)*f_Hq(ves)
        else:
            return 0
    
    def integrand2(z,x):
        '''
        Integrand of the Dark matter spike density considering
        an initial Hernquist density profile.

        Parameters
        ----------
        z: FLOAT
            Parameter relate with the angular momentum of the particles.
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Integrand of DM spike value dz. 
            Integrate over u values.
        '''
        return quad(integrand,0,1,args=(z,x),epsrel=1e-3)[0]
                                                         
    if r<=4:
        return 0                                                   
    # Change the parameter of the radius 
    x = (r*Rs)/a
    # Counter
    l = 0
    # Inizializate the empty arrays
    value = np.empty((n_step_z),dtype = float)
    # Chose a maximum value of z since 
    # for z=1 the integral diverge.np.linspace(0,maxz,n_step_z)
    maxz=0.995
    for z in np.linspace(0,maxz,n_step_z):
        # Obtain the integrand for each value of z
        value[l] = integrand2(z,x)
        l += 1
    #print(i)
    #print(end-start)
    z = np.linspace(0,maxz,n_step_z)
    # Interpolate the result of the integral over u
    interpolation = interpolate.interp1d(z,value)
    # Consider an array of possible values of z
    z1 = np.linspace(0,maxz,1000)
    # Integrate over u parameter
    inter = np.trapz(z1,interpolation(z1))
    # Obtain the DM density. 
    density = e_f_max(x)*(1/(np.sqrt(2)*(2*np.pi)**2*x))*(M_halo/a**3)*inter
    return density





def density_DMspike_sNFW(r,M_halo,M_bh,a,n_step_z,n_step_u):
    '''
    Dark matter spike density considering an initial DM sNFW
    density profile.
    Parameters
    ----------
    r : FLOAT
        Radius in which the density is evaluated.
        Must be in Rs units.
    M_halo : FLOAT
        Total mass of the initial halo.
        Must be in M_sun units.
    M_bh : FLOAT
        Black hole mass.
        Must be in M_sun units.
    a :  FLOAT
        Scale parameter of the Hernquist model.
    n_step_z : INT
        Number of steps to obtain the integral over z.
    n_step_u : INT
        Number of steps to obtain the integral over u.
    Returns
    -------
    FLOAT
    DM density value at radius 'r'.
    '''
    m = M_bh/M_halo
    G = ((const.G/const.c**4).to('cm/GeV')).value
    Rs = 2*G*M_bh
    
    def e_f_max(x):
        '''
        Maximum energy that a bound particle could have
        after the formation of the black hole considering
        the capture effect.
        
        Parameters
        ----------
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Maximum energy value.

        '''
        return (2*m/x)*(1-(8*m*G*M_halo/(x*a)))
    
    def L_max(e_f,x):
        '''
        Maximum angular momentum that a bound particle could have
        after the formation of the black hole.       

        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        x : FLOAT
            Dimensionless radius r/a.
            
        Returns
        -------
        FLOAT
            Maximum angular momentum value.

        '''
        return (2*x**2*(2*m/x-e_f))**(1/2)
    L_min = 4*m*(2*G*M_halo/a)**(1/2)

    def Irh(e_i,L):   
        '''
        Radial invariant action for the Hernquist density potential in
        dimensionless variables.

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        FLOAT
           Radial invariant action for the Hernquist potential.
        '''
        def integrand(x1):
            '''
              Radial invariant action integrand in dimensionless variables.
              Parameters
              ----------
              x1 : FLOAT
                    Radius.
             Returns
             -------
             FLOAT
               Radial invariant action integrand value.
            '''
            # Numerator of the integrand
            numerator = (4*x1**2-2*e_i*x1**2*(x1+1+(x1+1)**(1/2))-L**2*(x1+1+(x1+1)**(1/2)))
            denominator = (x1+1+(x1+1)**(1/2))*x1**2
            # Minimum value considered to avoid NaN errors
            if numerator<0 or denominator== 0:
                return 0
            
            return (numerator/((x1+1+(x1+1)**(1/2))*x1**2))**(1/2)
        # Obtain the limits
        lim = limits(e_i,L)
        return  np.sqrt(2)*quad(integrand,lim[0],lim[1],epsrel=1e-3)[0]
    def limits(e_i,L):
        '''
        Calculates the limits of the radial action integrand considering
        the initial sNFW density profile.

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        list
          List with the value of the two positive roots of the integrand.
          [x_-, x_+]

        '''
        # For obtaining the roots of the sNFW, we have used an approximate 
        # method. First, obtain the roots for the Hernquist which is easier and 
        # then, use this roots as initial point to find the roots in sNFW.
        coeff = [-2*e_i,2*(1-e_i),-L**2,-L**2]
        # Obtain the roots
        roots = np.roots(coeff)
        # Choose only the positive roots
        roots = roots[roots>=0]
        roots.sort()
        if any(np.iscomplex(roots)) or len(roots)<2:
            return [0,0]        
        if len(roots)>2:
            roots = [roots[1],roots[2]]
        else:
            roots =[roots[0],roots[1]]    
        # Once we have the roots of the Hernquist, we can obtain the roots of sNFW
        numerator= lambda x1 :(4*x1**2-2*e_i*x1**2*(x1+1+(x1+1)**(1/2))-L**2*(x1+1+(x1+1)**(1/2)))
        lims = np.array(optimize.fsolve(numerator, [roots[0], roots[1]]))
        lims.sort()
        return [lims[0],lims[1]]        
    def IrB(e_f,L):
        '''
        Radial adiabatic invariant action for the point mass potential in
        dimensionless variables.

        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        L : FLOAT
            Angular momentum.

        Returns
        -------
        FLOAT
           Radial invariant action for a point mass potential.
        '''
        return np.pi*(m/(2*e_f)**(1/2)-L)

        
    def e_f_to_e_i(e_f,L):
        '''
        Find the energy of the particles before the
        formation of the black hole e_i given the energy after 
        the formation of the black hole e_f.
        
        Parameters
        ----------
        e_f : FLOAT
            Energy after the formation of the black hole.
        L : FLOAT
            Angular momentum.


        Returns
        -------
        FLOAT
            Initial energy of the particles.

        '''
        def radial_action_conservation(e_i):
            '''
            Radial invariant action conservation,i.e., before and after 
            the formation of the black hole.

            Parameters
            ----------
            e_i : FLOAT
                Energy before the formation of the black hole.

            Returns
            -------
            FLOAT
                The difference between the radial actions.

            '''
            
            # Use this difference to only obtain values between
            #  0 and 1.
            # As we have used the Hernquist function of Ir_BH, we have to 
            # change the variables e_f and L.
            dif = Irh(e_i,L)-IrB(e_f/2,L/np.sqrt(2)) if e_i <= 1 and e_i > 0 else 1e10
            return dif
        # We minimize the difference to find the root. #
        return  optimize.brentq(radial_action_conservation,0,1)

    def f_SNFW_e(e_i):
       '''
        The phase-space distribution function of the sNFW model.
        # Article: https://arxiv.org/abs/1802.03349

        Parameters
        ----------
        e_i : FLOAT
            Energy before the formation of the black hole.

        Returns
        -------
        FLOAT
            Phase-space distribution function evaluate 
            in energy point e_i.

       '''    
       P1=-4*(32*e_i**6+416*e_i**5+1200*e_i**4-920*e_i**3-2198*e_i**2+399*e_i+504)
       P2=-8*(32*e_i**6+352*e_i**5+656*e_i**4-1176*e_i**3-586*e_i**2+173*e_i+360)
       P3=(e_i+8)*(128*e_i**5+512*e_i**4-576*e_i**3-480*e_i**2+56*e_i+171)
       return ((1/(7*2**(11/2)*(8-e_i)*(1-e_i)**2))*(252*((8+e_i
            )/(2*(1-e_i))**(1/2))*np.arcsin(e_i**(1/2)
            )+P1*(e_i/2)**(1/2)+P2*float(mp.ellipe(-e_i/8))+P3*float(
                mp.ellipk(-e_i/8))+189*(8+e_i)*float(mp.ellippi(e_i,-e_i/8))))
    
    def integrand(u,z,x):
        '''
        Integrand of the Dark matter spike density considering
        an initial sNFW density profile.

        Parameters
        ----------
        u : FLOAT
            Parameter relate with the energy of the particles.
        z: FLOAT
            Parameter relate with the angular momentum of the particles.
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Integrand of DM spike value dudz.

        '''
        if u != 0:
            e_f = u*e_f_max(x)
            L = (z*L_max(e_f, x)**2+(1-z)*L_min**2)**(1/2)   
            return f_SNFW_e(e_f_to_e_i(e_f,L))*((L_max(e_f,x)**2-L_min**2)/(1-z))**(1/2)
        else:
            return 0
    
    def integrand2(z,x):
        '''
        Integrand of the Dark matter spike density considering
        an initial sNFW density profile.

        Parameters
        ----------
        z: FLOAT
            Parameter relate with the angular momentum of the particles.
        x : FLOAT
            Dimensionless radius r/a.

        Returns
        -------
        FLOAT
            Integrand of DM spike value dz. 
            Integrate over u values.
        '''  
        # # Inicializate the array
        integrand_value = np.empty((n_step_u),dtype=float)
        # Select a value close to u=1 but not one since
        # it diverges with u=1.
        u = np.linspace(0,1-1e-10,n_step_u)
        i = 0
        # Calculate the integrand over u
        for u_value in u:
            integrand_value[i] = integrand(u_value,z,x)
            i += 1
        integrand_2= abs(np.trapz(u,integrand_value))
        # integrand_2
        return integrand_2 #quad(integrand,0,1,args=(z,x),epsrel=1e-3)[0]     
    if r<=4:
        return 0
    # Change the parameter of the radius 
    x = (r*Rs)/a
    # Counter
    l = 0
    # Inizializate the empty arrays
    value = np.empty((n_step_z),dtype = float)
    # Chose a maximum value of z since 
    # for z=1 the integral diverge. np.linspace(0,maxz,n_step_z)
    maxz=0.995
    for z in np.linspace(0,maxz,n_step_z):
        # Obtain the integrand for each value of z
        value[l] = integrand2(z,x)
        l += 1
    #print(i)
    #print(end-start)
    z = np.linspace(0,maxz,n_step_z)
    # Interpolate the result of the integral over u
    interpolation = interpolate.interp1d(z,value)
    # Consider an array of possible values of z
    z1 = np.linspace(0,maxz,1000)
    # Integrate over u parameter
    inter = np.trapz(z1,interpolation(z1))
    # Obtain the DM density. 
    density = e_f_max(x)*(3/(np.sqrt(2)*16*np.pi**2*x))*(M_halo/a**3)*inter
    return density


    

