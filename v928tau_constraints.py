from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

from routines_disk_constraints import vperitoe, etorperi, rhill

def create_region(x,y,color='grey'):
    X  = np.concatenate((x,np.flip(x)),0)
    Y  = np.concatenate((y[0],np.flip(y[-1])),0)
    XY = np.vstack((X,Y)).T
    region = Polygon(XY,facecolor=color,alpha=0.5)
    return region

def create_region2(x,y,color='grey'):
    bottom_mask = x[0] != np.nan
    bottom = np.vstack((x[0,bottom_mask],y[0,bottom_mask])).T
    for k in range(len(y)):
        #right = 
        print(k)
    return None

def create_limit(x,ylim,upper=False):
    X = np.concatenate((x,np.flip(x)),0)
    if upper == True:
        y0 = 100 * np.ones(len(x)) * ylim.unit
    else:
        y0 = np.zeros(len(x)) * ylim.unit
    y1 = ylim * np.ones(len(x))
    Y = np.concatenate((y0,y1),0)
    XY = np.vstack((X,Y)).T
    lim_region = Polygon(XY,facecolor='k',alpha=0.8)
    lim_region.set_hatch('/')
    return lim_region 

def set_prop_lim(prop,decimals=1):
    f = 10 ** decimals
    prop_min = np.floor(np.amin(prop).value*f)/f
    prop_max = np.ceil(np.amax(prop).value*f)/f
    return (prop_min, prop_max)

def plot_P_prop(P,Mp,prop,prop_name,prop_units='-',decimals=1,nlines=1,prop_lower=None,prop_upper=None):
    '''
    This function creates a plot showing a patch for the range of
    acceptable values and outlining it with the limits
    '''
    fig, ax = plt.subplots(1)
    fig.suptitle('%s Limits for Best Fit Model'%prop_name)
    ax.set_xlim(np.floor(Pmin.value), np.ceil(Pmax.value))
    ax.set_ylim(set_prop_lim(prop,decimals))
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('%s [%s]'%(prop_name,prop_units))
    # create and plot region
    P_prop = create_region(P,prop)
    ax.add_patch(P_prop)
    # outline region
    ax.plot(P,  prop[0], label='%.1f Mjup' % Mp[0].value)
    # add lines
    ind_incr = int((len(Mp)-1)/(nlines+1))
    for k in range(nlines):
        l = (k+1)*ind_incr
        ax.plot(P, prop[l], label='%.1f Mjup' % Mp[l].value)
    ax.plot(P, prop[-1], label='%.1f Mjup' % Mp[-1].value)
    # add limits
    if prop_upper != None:
        P_upper = create_limit(P,prop_upper,True)
        ax.add_patch(P_upper)
    if prop_lower != None:
        P_lower = create_limit(P,prop_lower)
        ax.add_patch(P_lower)
    ax.legend(loc='best')
    fig.show()
    return None

def plot_P_prop2(P,Mp,prop,prop_name,prop_units='-',decimals=1,nlines=1,prop_lower=None,prop_upper=None):
    '''
    This function creates a plot showing a patch for the range of
    acceptable values and outlining it with the limits
    '''
    fig, ax = plt.subplots(1)
    fig.suptitle('%s Limits for Best Fit Model'%prop_name)
    ax.set_xlim(np.floor(Pmin.value), np.ceil(Pmax.value))
    ax.set_ylim(set_prop_lim(prop,decimals))
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('%s [%s]'%(prop_name,prop_units))
    # create and plot region
    P_prop = create_region2(PP,prop)
    ax.add_patch(P_prop)
    # outline region
    ax.plot(P[0],  prop[0], label='%.1f Mjup' % Mp[0].value)
    # add lines
    ind_incr = int((len(Mp)-1)/(nlines+1))
    for k in range(nlines):
        l = (k+1)*ind_incr
        ax.plot(P[l], prop[l], label='%.1f Mjup' % Mp[l].value)
    ax.plot(P[-1], prop[-1], label='%.1f Mjup' % Mp[-1].value)
    # add limits
    if prop_upper != None:
        P_upper = create_limit(P,prop_upper,True)
        ax.add_patch(P_upper)
    if prop_lower != None:
        P_lower = create_limit(P,prop_lower)
        ax.add_patch(P_lower)
    ax.legend(loc='best')
    fig.show()
    return None

def plot_properties(E,RP,RH,A,Pmin,Pmax,Mp_min,Mp_max,contours=False):
    data = [[E,RP.value],[RH.value,A.value]] 
    title = [['Eccentricity [-]','Periastron Distance [AU]'],['Hill Radius [AU]','Semi-major Axis [AU]']]
    ext = (Pmin.value,Pmax.value,Mp_min.value,Mp_max.value)
    asp = (ext[1]-ext[0]) / (ext[3]-ext[2]) 
    fig,ax = plt.subplots(2,2) 
    fig.suptitle('Constraints on V928 tau based on Best Fit')
    for j in range(2): 
        for k in range(2):
            ax[j,k].set_title(title[j][k])
            ax[j,k].set_xlabel('Period [days]')
            ax[j,k].set_ylabel('Mass [Mjup]') 
            im = ax[j,k].imshow(data[j][k],origin='lower left',extent=ext,aspect=asp)
            fig.colorbar(im, ax=ax[j,k])
            if contours == True:
                dmin = np.nanmin(data[j][k])
                dmax = np.nanmax(data[j][k]) 
                lvls = np.linspace(dmin,dmax,7)
                C = ax[j,k].contour(data[j][k],levels=lvls,colors='r',extent=ext)
                ax[j,k].clabel(C, inline=1, fontsize=10) 
    fig.show()

def possible_P(Pmax,dP,n=1,lim=80): 
    Pu = (Pmax+dP)/n 
    Pm = Pmax/n 
    Pl = (Pmax-dP)/n
    if Pm > lim:
        Ps = np.array([Pl,Pm,Pu]) 
    else:
        Ps = None
    return Ps

def Period_mask(Pmax, dP):
    PS = np.zeros((0,3))
    n = 1
    cond = True
    while cond != False:
        Ps = possible_P(Pmax.value,dP.value,n) 
        if isinstance(Ps,np.ndarray):
            PS = np.concatenate((PS,Ps[None,:]),0)
        else:
            cond = False
        n += 1

    mask = 0
    for l in range(len(PS)): 
        mask += ((PP.value>PS[l,0])*(PP.value<PS[l,2])).astype(np.int)

    mask = ~mask.astype(np.bool)
    return mask




### User Defined Parameters
# maximum ratio of Rh/Rp, used to determine Mp_max
f_Rhp = 0.3
# load best fit parameters and unpack
filename = 'pb_newest.txt'
best_fit = np.loadtxt(filename)
r,tau,dy,dx,inc,tilt,ds = np.array([0.3572363968962847, 0.9961470648917765, 0.08917711202761658, 0.004254636522168245, 1.379670512617804, 1.623814087422141, 0.3000133832606969])
r,tau,dy,dx,inc,tilt,ds = best_fit

### Fixed Parameters
# V928tau A mass and radius
M1 = 0.60 * u.Msun
R1 = 1.56 * u.Rsun
# V928tau B mass and radius
M2 = 0.58 * u.Msun
R2 = 1.41 * u.Rsun
# V928tau A+B separation
asep = 32 * u.au
# eclipse time and depth
t_ecl = 0.5 * u.day
d_ecl = 0.6

### Derived Parameters
# minimum disk radius
Rd_min = R1 * np.sqrt(d_ecl)
# periods min from K2 data, max from 2nd eclipse
Pmin =  80 * u.day
Pmax = 575 * u.day
P = np.linspace(Pmin, Pmax, 5001)
# periastron velocity
vp = (2 * R1 / (ds * u.day)).to(u.km / u.s)
# mass of the planet
Mp_min =  0 * u.Mjup
Mp_max = ((3 * f_Rhp**3) / (1 - 3 * f_Rhp**3) * M1).to(u.Mjup)
Mp = np.linspace(Mp_min, Mp_max, 1001)

### Calculations
e  = vperitoe(M1, Mp[:,None], P[None,:], vp)
rp = etorperi(M1, Mp[:,None], vp, e)
rh = rhill(M1, Mp[:,None], rp)
a  = rp / (1 - e)

### Limits
Rd = ((r*u.day) * vp).to(u.au)
RH_min = 3 * Rd  # stability of the disk
A_max  = asep / 4. # stability of the orbit

plot_P_prop(P,Mp,e,'Eccentricity','-',2,1)
plot_P_prop(P,Mp,rh,'Hill Radius','AU',2,5,prop_lower=RH_min)
plot_P_prop(P,Mp,a,'Semi-major Axis','AU',2,1,prop_upper=A_max)


### GRIDS
MP,PP = np.mgrid[:len(Mp),:len(P)]
PP = PP * (Pmax - Pmin)/(len(P)-1.) + Pmin
MP = MP * (Mp_max - Mp_min)/(len(Mp)-1.) + Mp_min

### MASKS
mask_a  = a > A_max
mask_rh = rh < RH_min

### APPLYING MASKS
MP[mask_a]  = np.nan
MP[mask_rh] = np.nan
PP[mask_a]  = np.nan
PP[mask_rh] = np.nan

E  = vperitoe(M1, MP, PP, vp)
RP = etorperi(M1, MP, vp, E)
RH = rhill(M1, MP, RP)
A  = RP/(1-E)


plot_properties(E,RP,RH,A,Pmin,Pmax,Mp_min,Mp_max,True)


dP = 5*u.day

mask_p = Period_mask(Pmax,dP)

MP[mask_p] = np.nan
PP[mask_p] = np.nan

E  = vperitoe(M1, MP, PP, vp)
RP = etorperi(M1, MP, vp, E)
RH = rhill(M1, MP, RP)
A  =  RP/(1-E)

plot_properties(E,RP,RH,A,Pmin,Pmax,Mp_min,Mp_max,False)
