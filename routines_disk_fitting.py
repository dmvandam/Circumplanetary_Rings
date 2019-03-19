"""
This module contains all the routines and sub-routines necessary to
investigate a large parameter space of possible disk solutions for
a given light curve.

Input Parameters for the Phase Space Search are:
    te : duration of the eclipse
    dx : x-position of the centre of the disk
    dy : y-position of the centre of the disk
    f  : stretch factor of the disk

Limit Parameters
    slopes : of the ingress and egress
    R_hill : limits f per (dx,dy)

Output Parameters
    radii
    tilt
    inclination
    slope_ingress
    slope_egress

---------------------------------------------------------------------
Missing

Routine
    - convert tilt, inclination, and radius to a (te, dx, dy, f) point

Functionality 
    - find the max values of f for every (dx,dy)
    - find the range of tilt, inclination and gradient values

Plots
    - 3D surface plot?

"""
#####################################################################
######################## STANDARD MODULES ###########################
#####################################################################

import matplotlib.pyplot as plt
import numpy as np


#####################################################################
######################## DISK SUB ROUTINES ##########################
#####################################################################

def shear_ellipse_point(R, f, g, s, theta):
    '''
    Transforms a point on an ellipse (stretched in y, squeezed
    in x) to a point on a sheared ellipse.

    The input angle relates to the parametric form of a point 
    on a circle, x = R*cos(theta) and y = R*sin(theta). The 
    output point is the same point on a stretched, squeezed and
    sheared circle, in cartesian coordinates.

    Parameters
    ----------
    R : array_like (1-D)
        array containing all the radii for a face-on disk with
        impact parameter dy { R = np.hypot(te/2.,dy) }
    f : float
        the y stretch factor of the smallest disk.
    g : array_like (1-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(R**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx[None,:] / dy[:,None] }
    theta : float [rad]
        the angle of the point on the circle that will be transformed
        to the stretched, squeezed and sheared circle.

    Returns
    -------
    xp : array_like (2-D)
        the x-coordinate of the input point in the transformed circle.
    yp : array_like (2-D)
        the y-coordinate of the input point in the transformed circle.
    '''
    # x, y circle
    x = R[:,None] * np.cos(theta)
    y = R[:,None] * np.sin(theta)
    # xp, yp of stretched (f), squeezed (g), and sheared (s) circle
    yp = f * y
    xp = g[:,None] * x - s * yp
    return xp, yp

def theta_max_min(f, g, s):
    '''
    Determines the parametric angle of the location of either the
    semi-major axis or the semi-minor axis of an ellipse sheared 
    as follows:

    1) x = R*cos(th)
       y = R*sin(th)
    2) x' = g*x
       y' = f*y
    3) x" = x'-s*y'
       y" = y'
       
    g = te*f/(2*sqrt((Rf)^2-dy^2)), where R = sqrt(dy^2+(te/2)^2)
    and s = -dx/dy

    This is based on the fact that at the vertices and co-vertices
    of an ellipse dR"/dtheta = 0.

    Parameters
    ----------
    f : float
        the y stretch factor of the smallest disk.
    g : array_like (1-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(R**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx[None,:] / dy[:,None] }

    Returns
    -------
    theta_max_min : array_like (2-D) [rad]
        Array containing the angle of either the semi-major or semi-
        minor axis of an ellipse corresponding to the ellipse centre
        positions.

    Notes
    -----
    This function returns either the location of a co-vertex or a
    vertex (location of the semi-minor or semi-major axis, resp.).

    The two are separated by pi/2 radians.
    '''
    # numerator and denominator theta_max_min = tan(2*theta) = num / den
    theta_num = 2 * f * g[:,None] * s
    theta_den = (s**2 + 1) * f**2 - g[:,None]**2
    # theta_max_min
    theta_max_min = 0.5 * np.arctan2(theta_num, theta_den)
    return theta_max_min

def find_ellipse_parameters(R, f, g, s):
    '''
    Finds the semi-major axis, a, semi-minor axis, b, the tilt and 
    the inclination of the smallest ellipse stretched by, f, that 
    is centred at (dx,dy) w.r.t. the centre of the eclipse with 
    duration te.

    Parameters
    ----------
    R : array_like (1-D)
        array containing all the radii for a face-on disk with
        impact parameter dy { R = np.hypot(te/2.,dy) }
    f : float
        the y stretch factor of the smallest disk.
    g : array_like (1-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(R**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx[None,:] / dy[:,None] }

    Returns
    -------
    a : array_like (2-D)
        Array containing all the semi-major axes of the ellipses
        investigated. i.e. with their centres at (dx,dy).
    b : array_like (2-D)
        Array containing all the semi-minor axes of the ellipses
        investigated. i.e. with their centres at (dx,dy).
    tilt : array_like (2-D)
        Array containing all the tilt angles of the ellipses
        investigated. i.e. with their centres at (dx,dy). Tilt is
        the angle of the semi-major axis w.r.t. the x-axis. [deg]
    inclination : array_like (2-D)
        Array containing all the inclination angles of the ellipses
        investigated. i.e. with their centres at (dx,dy). 
        Inclination is based on the ratio of semi-minor to semi-major
        axis. [deg]
    '''
    # find position of (co-) vertices
    theta1 = theta_max_min(f, g, s)
    theta2 = theta1 + np.pi/2
    x1, y1 = shear_ellipse_point(R, f, g, s, theta1)
    x2, y2 = shear_ellipse_point(R, f, g, s, theta2)
    # find the semi-major and semi-minor axes
    R1 = np.hypot(x1,y1)
    R2 = np.hypot(x2,y2)
    a = np.maximum(R1,R2)
    b = np.minimum(R1,R2)
    # determine the inclination
    inclination = np.arccos(b/a)
    # determine the tilt
    tilt = np.arctan2(y1,x1) # assuming R1 > R2
    tilt_mask = R2 > R1 # find where above is not true
    tilt = tilt + tilt_mask*np.pi/2 # at above locations add np.pi/2
    return a, b, np.rad2deg(tilt), np.rad2deg(inclination)

def find_ellipse_slopes(te, dx, dy, f, g, s):
    '''
    Finds the slopes of the tangents of the given ellipse centred at
    (dx,dy) at the height of the eclipse.
    
    i.e. finds the tangents of the ellipse at (-te/2, 0) and 
    (te/2, 0), which is (-te/2-dx,-dy) and (te/2-dx,-dy) in
    the frame that coincides with the centre of the ellipse.

    This is converted to an angle so that the range is [0,1].

    Parameters
    ----------
    te : float
        duration of the eclipse.
    dx : array_like (1-D)
        array containing all the x-shifts of the disk centre
    dy : array_like (1-D)
        array containing all the impact parameters of the disks
        investigated.
    f : float
        the y stretch factor of the smallest disk.
    g : array_like (1-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(R**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx[None,:] / dy[:,None] }
    
    Returns
    -------
    slope1 : array_like (2-D)
        Array containing all the slopes of the tangents of the left
        hand side. i.e. at (-te/2,-dy).
    slope2 : array_like (2-D)
        Array containing all the slopes of the tangents of the right
        hand side. i.e. at (te/2,-dy). Note that the slopes are
    '''
    # determining coordinates on circle
    y = -dy
    x1 = -te/2. - dx
    x2 =  te/2. - dx
    # slopes are based on:
    # Rp**2 = xp**2 + yp**2 = (g*x - s*f*y)**2 + (f*y)**2
    # dy/dx = (s*f*g*y - g^2*x) / ((s**2 + 1)f**2*y - s*f*g*x)
    # left side
    dy1 = -s * f**2 * y[:,None] - f**2 * x1[None,:]
    dx1 = (s**2 * f**2 + g[:,None]**2) * y[:,None] + s * f**2 * x1[None,:]
    slope1 = dy1/dx1
    # right side
    dy2 = -s * f**2 * y[:,None] - f**2 * x2[None,:]
    dx2 = (s**2 * f**2 + g[:,None]**2) * y[:,None] + s * f**2 * x2[None,:]
    slope2 = dy2/dx2
    return slope1, slope2


#####################################################################
################# QUADRANT EXPANSION SUB ROUTINES ###################
#####################################################################

def reflect_quadrants(array):
    '''
    Input array is the top right quadrant (QI). This is reflected
    across the relevant axes to create a four quadrant array. This
    can be used for radius (as this is quadrant independent) and
    inclination (also quadrant independent).

    Parameters
    ----------
    array : array_like (2-D)
        m x n array containing for example the semi-major axis, the,
        semi-minor axis,or the inclination of the ellipses.

    Returns
    -------
    new_array : array_like (2-D)
        2m x 2n array containing same information as above but
        flipped and reflected to fill up four quadrants

    Example
    -------
    array = np.array([[0,1],[1,1]])
    new_array = reflect_quadrants(array)

             [1,1,1,1]
    [0,1] -> [1,0,0,1]
    [1,1] -> [1,0,0,1]
             [1,1,1,1]

    Notes
    -----
    Above example may seem confusing, but this is because of the way
    python indices work, y is increasing from top to bottom so QI is
    at the bottom right, instead of top right, and is reflected w.r.t.
    the y-axis
    '''
    # create new array
    dy, dx = array.shape
    new_array = np.zeros((2*dy, 2*dx))
    # reflections and copying
    new_array[:dy,:dx] = np.flipud(np.fliplr(array)) # lower left
    new_array[:dy,dx:] = np.flipud(array) # lower right
    new_array[dy:,:dx] = np.fliplr(array) # upper left
    new_array[dy:,dx:] = array # upper right (original quadrant)
    return new_array

def reflect_tilt(array):
    '''
    Input array is the top right quadrant (QI). This is reflected
    across the relevant axes to create a four quadrant array. This
    can be used for tilt, which IS quadrant dependent. Tilt is assumed
    to be from 0 - 180 degrees.

    Parameters
    ----------
    array : array_like (2-D)
        m x n array containing for example the semi-major axis, the,
        semi-minor axis, the inclination or the tilt of the ellipses.

    Returns
    -------
    new_array : array_like (2-D)
        2m x 2n array containing same information as above but
        flipped and reflected to fill up four quadrants

    Example
    -------
    array = np.array([[30,15],[15,15]])
    new_array = reflect_tilt(array)

               [ 15, 15,165,165]
    [30,15] -> [ 15, 30,150,165]
    [15,15] -> [165,150, 30, 15]
               [165,165, 15, 15]

    Notes
    -----
    Above example may seem confusing, but this is because of the way
    python indices work, y is increasing from top to bottom so QI is
    at the bottom right, instead of top right, and is reflected w.r.t.
    the y-axis
    '''
    # create new array
    dy, dx = array.shape
    new_array = np.zeros((2*dy, 2*dx))
    # reflections and copying
    new_array[:dy,:dx] = np.flipud(np.fliplr(array)) # lower left
    new_array[:dy,dx:] = np.flipud(180-array) # lower right
    new_array[dy:,:dx] = np.fliplr(180-array) # upper left
    new_array[dy:,dx:] = array # upper right (original quadrant)
    return new_array

def reflect_slopes(slope_left, slope_right):
    '''
    Input array is the top right quadrant (QI) of the ellipse slopes
    on the left and on the right. The ellipse is reflected across the
    relevant axes, which means that the slope_left and slope_right
    arrays are flipped (about an axis), copied and switched (i.e.
    slope left can be found in the new_slope_right and vice versa)
    to create a four quadrant array.

    Parameters
    ----------
    slope_left : array_like (2-D)
        m x n array containing the left hand slopes of the ellipses
        investigated.
    slope_right : array_like (2-D)
        m x n array containing the right hand slopes of the ellipses
        investigated.

    Returns
    -------
    new_slope_left : array_like (2-D)
        2m x 2n array filled up as four quadrants using the necessary
        flipping, copying and switching.
    new_slope_right : array_like (2-D)
        2m x 2n array filled up as four quadrants using the necessary
        flipping, copying and switching.

    Example
    -------
    slope_left = np.array([[0,1],[1,1]])
    slope_right = np.array([[1,1],[1,1]])
    new_slope_left, new_slope_right = reflect_slopes(slope_left,slope_right)

                    [ 1, 1,-1,-1]   [ 1, 1,-1,-1]
    [0,1] [1,1]  -> [ 1, 1, 0,-1]   [ 1, 0,-1,-1]
    [1,1],[1,1]  -> [-1,-1, 0, 1]   [-1, 0, 1, 1]
                    [-1,-1, 1, 1] , [-1,-1, 1, 1]

    Notes
    -----
    Above example may seem confusing, but this is because of the way
    python indices work, y is increasing from top to bottom so QI is
    at the bottom right, instead of top right, and is reflected w.r.t.
    the y-axis.
    '''
    # determining the shape
    dy,dx = slope_left.shape
    # creating new arrays
    new_slope_left = np.zeros((2*dy, 2*dx))
    new_slope_right = np.zeros((2*dy, 2*dx))
    # below U is upper, L is lower, l is left,r is right
    # reflections, copying and switching - LEFT
    new_slope_left[dy:,dx:] = slope_left # Ur (original quadrant)
    new_slope_left[dy:,:dx] = np.fliplr(-slope_right) # Ul (switched)
    new_slope_left[:dy,dx:] = np.flipud(-slope_left) # Lr
    new_slope_left[:dy,:dx] = np.flipud(np.fliplr(slope_right)) # Ll
    # reflections, copying and switching - RIGHT
    new_slope_right[dy:,dx:] = slope_right  
    new_slope_right[dy:,:dx] = np.fliplr(-slope_left)
    new_slope_right[:dy,dx:] = np.flipud(-slope_right)
    new_slope_right[:dy,:dx] = np.flipud(np.fliplr(slope_left))
    return new_slope_left, new_slope_right

#####################################################################
####################### MAIN DISK ROUTINES ##########################
#####################################################################

def investigate_ellipses(te, xmax, ymax, f, nx=50, ny=50, ymin=1e-8, xmin=1e-8):
    '''
    Investigates the full parameter space for an eclipse of duration
    te with centres at [-xmax,xmax] (2*nx), [-ymax,ymax] (2*ny)

    Parameters
    ----------
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    f : float
        the y stretch factor of the smallest disk.
    nx : integer
        number of steps to investigate in the x direction. i.e.
        investigated space is np.linspace(-xmax, xmax, 2*nx).
    ny : integer
        number of steps to investigate in the y direction. i.e.
        investigated space is np.linspace(-ymax, ymax, 2*ny).
    ymin : float
        this is necessary because the shear parameter s = -dx/dy
        so dy != 0
    xmin : float
        this is to prevent that the middle two columns of the final
        grid actually show the same information

    Returns
    -------
    a : array_like (2-D)
        Array containing the semi-major axes of the investigated
        ellipses.
    b : array_like (2-D)
        Array containing the semi-minor axes of the investigated
        ellipses.
    tilt : array_like (2-D)
        Array containing the tilt angles [deg] of the investigated
        ellipses. Tilt is the angle of the semi-major axis w.r.t.
        the x-axis
    inclination : array_like (2-D)
        Array containing the inclination angles [deg] of the
        investigated ellipses. Inclination is the angle obtained
        from the ratio of the semi-minor to semi-major axis
    slope_left : array_like (2-D)
        Array containing all the slopes of the left ellipse edge at
        the eclipse height. Note that this slope is defined as the
        absolute value of the sine of the arctangent of dy/dx.
    slope_right
        Array containing all the slopes of the right ellipse edge at
        the eclipse height. Note that this slope is defined as the
        absolute value of the sine of the arctangent of dy/dx.

    Notes
    -----
    This function investigates the phase space available and the
    returned arrays can be used to make plots, gain insight and
    boundaries can be applied to these grids to limit the number of
    valid solutions for the given eclipse profile.
    '''
    # creating phase space
    dy = np.linspace(0, ymax, ny)
    dy[0] = ymin
    dx = np.linspace(0, xmax, nx)
    # important details
    R = np.hypot(te/2.,dy)
    s = -dx[None,:] / dy[:,None]
    # squeeze factor g related to stretch factor f
    g = (te * f) / (2 * np.sqrt(R**2 * f**2 - dy**2))
    # to prevent numerical errors
    if f == 1:
        g = np.ones_like(R)
    # investigating phase space
    a, b, tilt, inclination = find_ellipse_parameters(R, f, g, s)
    slope_left, slope_right = find_ellipse_slopes(te, dx, dy, f, g, s)
    # filling the quadrants
    radii = reflect_quadrants(a)
    tilt = reflect_tilt(tilt)
    inclination = reflect_quadrants(inclination)
    slope_left, slope_right = reflect_slopes(slope_left, slope_right)
    # converting to from angles to gradients
    grad_left = np.abs(np.sin(np.arctan2(slope_left,1)))
    grad_right = np.abs(np.sin(np.arctan2(slope_right,1)))
    return radii, tilt, inclination, grad_left, grad_right


#####################################################################
######################### BOUND FUNCTIONS ###########################
#####################################################################

def mask(mask_array, lower_limit, upper_limit, arrays):
    '''
    This function creates a lower and upper limit mask from mask_array
    and applies this to all the arrays in arrays (mask_array should be
    in arrays).

    Parameters
    ----------
    mask_array : array_like (2-D)
        Array containing values on which to apply limits
    lower_limit : float
        lower limit of mask_array that is considered acceptable.
    upper_limit : float
        upper limit of mask_array that is considered acceptable.
    arrays : list of array_like
        arrays (including mask_array) that will be masked

    Returns
    -------
    arrays : list of array_like
        arrays (including mask_array) that have been masked according
        to the lower and upper limits imposed.

    Notes
    -----
    masked out values are converted to np.nan (since 0 is a value that
    contains information)
    '''
    # creating mask
    mask_lower = mask_array > lower_limit
    mask_upper = mask_array < upper_limit
    mask_total = mask_lower * mask_upper
    # applying mask
    for i in range(len(arrays)):
        # makes masked out = 0
        arrays[i] = arrays[i] * mask_total
        # for some arrays 0 is a valid option so make 0's nans
        arrays[i][mask_total == False] = np.nan
    return arrays


#####################################################################
######################### PLOTTING ROUTINE ##########################
#####################################################################

def plot_property(disk_prop, prop_name, te, xmax, ymax, f, lvls="n", vmin=0, vmax=1e8, root=''):
    '''
    Plots a geometrical property of the investigated ring systems.

    Parameters
    ----------
    disk_prop : array_like (2-D)
        array containing a disk property for all the investigated
        ellipses.
    prop_name : string
        string containing the name of the property
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    f : float
        the y stretch factor of the smallest disk.
    lvls : str, float, array_like (1-D) [default = "n"]
        either nothing specified (str), number of levels (float)
        or the specific levels (array_like [1-D]).
    vmin : float [default = 0]
        minimum value in the colourmap.
    vmax : float [default = 1e8]
        maximum value in the colourmap.
    root : string [default = '']
        string containing the path where the figure will be saved.
    title : string [default = '']

    Returns
    -------
    fig : root + "te_%.3f_f_%.3f_%s.png" % (te, f, prop_name)
    '''
    ext = (-xmax,xmax,-ymax,ymax)
    title = prop_name.capitalize()
    save  = prop_name.lower()
    # create figure
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('x offset [days]')
    plt.ylabel('y offset [days]')
    plt.title('%s of Disk with $f$ = %.3f and $t_e$ = %.3f'%(title, f, te))
    # contours
    if lvls == "n":
        c = plt.contour(disk_prop, colors='r', extent=ext)
    elif isinstance(lvls,np.int):
        levels = np.linspace(vmin, vmax, lvls+1)[1:]
        c = plt.contour(disk_prop, levels=levels, colors='r', extent=ext)
    else:
        c = plt.contour(disk_prop, lvls, colors='r', extent=ext)
    plt.gca().clabel(c, c.levels, inline=True, fmt='%.1f', fontsize=8)
    # plot data
    plt.imshow(disk_prop, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower left', extent=ext)
    plt.colorbar()
    fig.savefig(root+'te_%.3f_f_%.3f_%s.png'%(te, f, save))
    return None
