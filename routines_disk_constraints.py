import numpy as np
import astropy.units as u
import astropy.constants as c

@u.quantity_input
def atoP(a:u.au, m1:u.M_sun, m2:u.M_jup)->u.year:
    """
    Calculate period from orbital radius and masses

    Args:
        a: semi-major axis
        m1, m2: Primary and secondary masses

    Returns:
        P: orbital period

    >>> import astropy.units as u
    >>> atoP(1.*u.au, 1.0*u.M_sun, 1.0*u.M_jup)
    <Quantity 0.99954192 yr>
    """

    # a^3/P^2 = (G/4pipi) (m1 + m2)

    const = c.G/(4.*np.pi*np.pi)

    mu = m1 + m2

    P2 = np.power(a,3.)/(const*mu)

    P = np.power(P2, 0.5)
    return P

@u.quantity_input
def Ptoa(P:u.year, m1:u.M_sun, m2:u.M_jup)->u.au:
    """calculate orbital radius from period

    Args:
        P: orbital period
        m1, m2: Primary and secondary masses

    Returns:
        a: semi-major axis

    >>> import astropy.units as u
    >>> Ptoa(11.86*u.year, 1.0*u.M_sun, 1.0*u.M_jup)
    <Quantity 5.20222482 AU>
    """

    # a^3/P^2 = (G/4pipi) (m1 + m2)
    const = c.G / (4.*np.pi*np.pi)
    mu = m1 + m2
    a3 = P*P*const*mu
    aa = np.power(a3, 1./3.)

    return aa

@u.quantity_input
def vcirc(m1:u.Msun,m2:u.Mjup,a:u.au)->u.km/u.s:
    """
    Circular orbital velocity of m2 about m1 at distance a

    Args:
        m1, m2: Primary and secondary masses
        a: semimajor axis

    Returns:
        velocity: circular orbital velocity

    >>> import astropy.units as u
    >>> vcirc(1.0 *u.M_sun, 1.0 * u.M_jup, 5.2 * u.au)
    <Quantity 13.06768412 km / s>
    """

    # http://en.wikipedia.org/wiki/Circular_orbit
    mu = c.G * (m1 + m2)
    vcirc = np.power(mu /a, 0.5)
    return vcirc

@u.quantity_input
def vperitoe(m1:u.Msun,m2:u.Mjup,P:u.year,vperi:u.km/u.s):
    """
    Finds the eccentricity necessary to get a periastron
    velocity (vperi) for given masses and Period

    Args:
        m1: Primary mass
        m2: Secondary mass
        P:  orbital period
        vperi: periastron velocity

    Returns:
        e:  eccentricity
    """
    a = Ptoa(P,m1,m2)
    mu = c.G * (m1 + m2)
    x = a*vperi**2/mu
    e = (x.decompose() - 1) / (x.decompose() + 1)
    return e

@u.quantity_input
def etovperi(m1:u.Msun,m2:u.Mjup,P:u.year,e)->u.km/u.s:
    """
    Elliptical maximum velocity

    Args:
        m1: Primary and secondary masses
        P: orbital period
        e: eccentricity [0,1)

    Returns:
        velocity: maximum velocity in elliptical orbit

    >>> import astropy.units as u
    >>> vmaxellip(1.0 *u.M_sun, 1.0 * u.M_earth, 1.0 * u.yr, 0.0)
    <Quantity 29.78490916 km / s>
    """
    mu =  c.G * (m1 + m2)
    c1 = 2 * np.pi * mu / P
    c2 = (1 + e) / (1 - e)

    vmax = np.power(c1, 1./3.) * np.power(c2, 1./2.)
    # http://en.wikipedia.org/wiki/Circular_orbit
    return vmax

@u.quantity_input
def etoP(m1:u.Msun,m2:u.Mjup,vperi:u.km/u.s,e)->u.yr:
    '''
    finds the period of an orbit for m1 and m2 given
    vperi and e
    Args:
        m1: Primary mass
        m2: Secondary mass
        vperi: the desired periastron velocity
        e: eccentricity of the orbit

    Returns:
        P: period of the orbit in days
    '''
    mu = c.G*(m1+m2)
    c2 = (1+e)/(1-e)
    P = 2*np.pi*mu/(vperi**2/c2)**(3/2.)
    return P

@u.quantity_input
def etorperi(m1:u.Msun,m2:u.Mjup,vperi:u.km/u.s,e)->u.au:
    '''
    finds the periastron distance for m1 and m2 given
    vperi and e
    
    Args:
        m1: Primary mass
        m2: Secondary mass
        vperi: the periastron velocity
        e: eccentricity of the orbit

    Returns:
        rperi: the periastron distance
    '''
    mu = c.G*(m1+m2)
    rperi = mu*(1+e)/vperi**2
    return rperi

@u.quantity_input
def rhill(m1: u.Msun, m2: u.Mjup, a: u.au)->u.au:
    """
    Hill radius of the secondary m2 orbiting around m1
    
    Args:
        m1, m2: primary and secondary masses
        a: distance between m1 and m2

    Returns:
        rhill: radius of Hill sphere of m2

    >>> import astropy.units as u
    >>> rhill(1.0 * u.M_sun, 1.0 * u.M_jup, 5.2 * u.au)
    <Quantity 0.35489325 AU>
    """

    mu = m2 / (m1 + m2)
    rh = a * np.power(mu/3., 1./3.)
    return rh

@u.quantity_input
def rdisktommin(m1:u.Msun,rperi:u.au,rdisk:u.au)->u.Mjup:
    '''
    gives the minimum mass of a secondary to bind a disk with
    rdisk if the condary is separated from m1 by rperi

    Args:
        m1: Primary mass
        rperi: distance at periastron
        rdisk: size of the disk

    Returns: 
        mmin: minimum mass of the secondary to have an rdisk
              at rperi
    '''
    rh = rdisk/0.3
    mmin = 3*m1*(rh/rperi)**3
    return mmin

@u.quantity_input
def rdiskmin(r1:u.Rsun,eclipse_depth)->u.au:
    '''
    This function calculates the minimum disk size to create an eclipse
    depth 
    
    Args:
        r1: radius of the primary
        eclipse_depth: the fraction of light blocked

    Returns:
        rdisk: radius of smallest disk required to block this fraction
               of light, this is a face on concentric disk
    '''
    rdisk = r1*np.sqrt(eclipse_depth)
    return rdisk

if __name__=="__main__":
    def print_lines(n):
        for z in range(n):
            print('')
        return None
    print_lines(2)
    import matplotlib.pyplot as plt
    # v928 tau parameters
    M1 = 0.60 * u.M_sun
    M2 = 0.58 * u.M_sun
    R1 = 1.56 * u.R_sun
    R2 = 1.41 * u.R_sun
    asep = 32.0 * u.au
    print('V 928tau consists of 2 stars of mass {0}, {1}, with radii {2}, {3} and a separation of {4}'.format(M1,M2,R1,R2,asep))

    # photometric data
    Pmin = 80.0 * u.day
    te   = 0.50 * u.day
    ts   = 2./3. * te
    print('the length of the data and thus P_min is {0} with an eclipse duration, t_e, of {1} with a depth of {2}. Due to the fact that the star is convolved with the disk we assume that the size of the star in days must be 2/3 of t_e, thus {3}'.format(Pmin,te,0.60,ts))
    print('')
    # let's make a grid for calculations
    # P from Pmin to Pmax
    Pmin = Pmin.to(u.yr)
    amax = asep / 4.
    Pmax = atoP(amax,M1,0.0*u.M_jup) * 0 + 1.35*u.yr
    # calculations for planet
    Mpmin = 0.0 * u.M_jup
    Mpmax = 60.0 * u.M_jup
    #Mpmax = (3*0.3**3)/(1-3*0.3**3) * M1.to(u.M_jup)

    M,P = np.mgrid[:1000,:10000]
    P = P*(Pmax-Pmin)/999.+Pmin
    M = M*(Mpmax-Mpmin)/999.+Mpmin
    
    E = np.linspace(0,1,11)[:-1]

    vperi = etovperi(M1,M[:,:,None],P[:,:,None],E[None,None,:])
    Rmindisk = ((ts/2.)*vperi).to(u.au)
    Rperi = etorperi(M1,M[:,:,None],vperi,E[None,None,:])
    RH = rhill(M1,M[:,:,None],Rperi)
    Zi = Rmindisk/RH
    for i,e in enumerate(E):
        fig,ax = plt.subplots(1,2)
        fig.suptitle('Mass Period Relations with Eccentricity = %.3f'%e)
        Zi[:,:,i][(RH[:,:,i]/Rperi[:,:,i] > 0.3)] = np.nan
        Zi[:,:,i][(RH[:,:,i]<R1)] = np.nan
        Zi[:,:,i][((R1/vperi[:,:,i]).to(u.day) > te/2.)] = np.nan
        #ax[0].imshow(Zi[:,:,i],extent=(Pmin.value,Pmax.value,Mpmin.value,Mpmax.value),origin='lower left',cmap='viridis',vmax=0.31)
        title = ['Zi','stellar radius in days','Periastron Velocity','min Rdisk']
        data = [Zi[:,:,i], (R1/vperi[:,:,i]).to(u.day).value]
        vmax = [0.3,te.value/2.,None]
        lvls = [0.05,0.15,0.25]
        for j in range(2):
            ax[j].set_xlabel('Period [yrs]')
            ax[j].set_ylabel('Mass [Mjup]')
            ax[j].set_title(title[j])
            im = ax[j].imshow(data[j],extent=(Pmin.value,Pmax.value,Mpmin.value,Mpmax.value),origin='lower left',cmap='viridis',vmax=vmax[j],aspect='auto')
            c = ax[j].contour(data[j],extent=(Pmin.value,Pmax.value,Mpmin.value,Mpmax.value),levels=lvls,colors='r')
            ax[j].clabel(c, c.levels, inline=True, fmt='%.2f', fontsize=8)
            plt.colorbar(im,ax=ax[j])
        fig.show()
        
    '''
    amin = Ptoa(Pmin,M1,Mp)
    vc_max = vcirc(M1,Mp,amin)
    print('if M_p = {0} then for a P_min = {1}, a_min = {2} and thus vc_max = {3}'.format(Mp,Pmin,amin,vc_max))
    R1t = (R1/vc_max).to(u.day)
    print('there is a problem because R_1 in days is {0} and t_e is {1} (< 2 R_1), this means we need a velocity larger than vc_max, which obtain through having a periastron passage of an eccentric orbit'.format(R1t,te))
    print('')
    amax = asep / 4.
    Pmax = atoP(amax,M1,Mp).to(u.yr)
    print("before we do, we set up some constraints, for example on the maximum a between planet and star, we take the separation between the two stars and divide by 4 to get a_max = {0}, P_max = {1}, again assuming M_p = {2}".format(amax,Pmax,Mp))
    # finding eccentricity
    vperi = 2*(R1 / te).to(u.km/u.s)
    #e = etovperi(M1.to(u.kg), Mp.to(u.kg), Pmin.to(u.s), vperi)
    e = vperitoe(M1, Mp, Pmin, vperi)
    print('')
    print('the minimum eccentricity to ensure that vperi = {0} and thus R_1 = {1} is e = {2}'.format(vperi,te,e))
    # independent
    E = np.linspace(e,1,10001)[:-1]
    # dependent Period and Hill sphere
    P = etoP(M1,Mp,vperi,E)
    Rp = etorperi(M1,Mp,vperi,E)
    Rh = rhill(M1,Mp,Rp)
    Rhtol = 1e-6*u.au
    # want to find Mmax
    atlim = np.ones_like(E).astype(np.bool) #atlim shows whether
    steps = np.ones_like(E)
    Mmax = np.zeros_like(E)*u.M_jup
    step_size = M1/2.
    counter = 0
    print('finding max mass of secondary')
    while np.sum(atlim) != 0:
        counter+=1
        print('    trial %02i, sum at lim = %i'%(counter,np.sum(atlim)))
        if len(np.unique(steps))> 1:
            print(steps)
        Mmax += steps*step_size
        P = etoP(M1,Mmax,vperi,E)
        Rp = etorperi(M1,Mmax,vperi,E)
        Rh = rhill(M1,Mmax,Rp)
        steps[Rh > 0.3*Rp] = -1
        steps[Rh < 0.3*Rp] =  1
        steps[(Rh>0.3*Rp-Rhtol)*(Rh<0.3*Rp+Rhtol)] = 0
        atlim[steps==0] = 0
        step_size /= 2
    #rdisk = 0.01*u.au
    #print('finding minimum mass of secondary for a disk size of {0}'.format(rdisk))
    #mmin = rdisktommin(M1,Rp,rdisk)
    print('now we start plotting some things and show their constraints')
    emin = np.copy(e)
    emax = E[np.argmin(np.abs(P-Pmax))]
    rdmin = rdiskmin(R1,0.6)
    mmin = rdisktommin(M1,Rp,rdmin)
    # E plots
    fig,ax = plt.subplots(2,2)
    fig.suptitle('the effect of eccentricity')
    ax[0,0].set_xlabel('log$_{10}$ (Period [yrs])')
    ax[0,0].set_ylabel('eccentricity [-]')
    ax[0,0].plot(np.log10(P.value),E)
    ax[0,0].axvline(x=np.log10(Pmin.to(u.yr).value))
    ax[0,0].axvline(x=np.log10(Pmax.value))
    ax[0,0].axhline(y=emin)
    ax[0,0].axhline(y=emax)
    ax[1,0].plot(E,Rh)
    fig.show()
    Pl = np.log10(P.value)
    fig,ax1 = plt.subplots()
    ax1.set_xlabel('log$_{10}$(Period [yr])')
    ax1.set_ylabel('$M_p$ [$M_{jup}$]')
    ax2 = ax1.twinx()
    ax1.plot(Pl,mmin,label='$M_{min}$, $r_{disk}$ = %.3f %s'%(rdmin.value,rdmin.unit))
    ax1.plot(Pl,Mmax,label='$M_{max}$, $r_{disk}$ = 0.3 $R_{hill}$')
    ax1.axvline(x=np.log10(Pmin.to(u.yr).value))
    ax1.axvline(x=np.log10(Pmax.value))
    ax2.set_ylabel('$M_p$ [$M_{earth}$]')
    ax2.plot(Pl,mmin.to(u.M_earth),label='$M_{min}$, $r_{disk}$ = %.3f %s'%(rdmin.value,rdmin.unit))
    ax2.plot(Pl,Mmax.to(u.M_earth),label='$M_{max}$, $r_{disk}$ = 0.3 $R_{hill}$')
    plt.legend()
    fig.show()
    print_lines(2)
    '''
