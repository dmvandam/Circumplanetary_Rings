"""
This module contains all the routines and sub-routines necessary to
investigate a large parameter space of possible disk solutions for
a given light curve.

Input Parameters for the Phase Space Search are:
    te : duration of the eclipse
    dx : x-position of the centre of the disk
    dy : y-position of the centre of the disk
    f  : the y stretch factor

Limit Parameters
    slopes : of the ingress and egress
    R_hill : limits f per (dx,dy)

Output Parameters
    radii
    tilt
    inclination
    slope_ingress
    slope_egress

Note this function can also find the max values of f for each point
numerically, and determine how disk parameters vary with f.

---------------------------------------------------------------------
Missing

Routine
    - convert tilt, inclination, and radius to a (te, dx, dy, f) point

Functionality 
    - proper way to model / plot these limits

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

def shear_ellipse_point(Rmin, f, g, h, s, theta):
    '''
    Transforms a point on an ellipse (stretched in y, squeezed
    in x) to a point on a sheared ellipse.

    The input angle relates to the parametric form of a point 
    on a circle, x = Rmin*cos(theta) and y = Rmin*sin(theta). The 
    output point is the same point on a stretched, squeezed and
    sheared circle, in cartesian coordinates.

    Parameters
    ----------
    Rmin : array_like (2-D [:,None])
        array containing all the radii for a face-on disk with
        impact parameter dy { Rmin = np.hypot(te/2.,dy) }
    f : float, array_like (2-D)
        the y stretch factor of the smallest disk.
    g : float, array_like (2-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(Rmin**2 * f**2 - dy**2)) }
    h : float, array_like (2-D)
        array containing all the R growth factors neccesary to
        compensate for a stretch factor f < 1
        { h = np.hypot(te / 2., dy / f) / Rmin } 
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }
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
    x = Rmin * np.cos(theta)
    y = Rmin * np.sin(theta)
    # x, y after stretch (f), squeeze (g), grow (h) & shear (s)
    yp = f * h * y
    xp = g * h * x - s * yp
    return xp, yp

def theta_max_min(f, g, s):
    '''
    Determines the parametric angle of the location of either the
    semi-major axis or the semi-minor axis of an ellipse sheared 
    as follows:

    1) x  = Rmin * np.cos(theta)
       y  = Rmin * np.sin(theta)
    2) x' = g * x
       y' = f * y
    3) x" = x' - s * y'
       y" = y'
       
    g = te * f / (2 * sqrt( (Rminf)^2 - dy^2)), where 
    Rmin = sqrt(dy^2+(te/2)^2) and s = -dx/dy

    This is based on the fact that at the vertices and co-vertices
    of an ellipse dR"/dtheta = 0.

    Parameters
    ----------
    f : float, array_like (2-D)
        the y stretch factor of the smallest disk.
    g : float, array_like (2-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(Rmin**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }

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
    theta_num = 2 * f * g * s
    theta_den = (s**2 + 1) * f**2 - g**2
    # theta_max_min
    theta_max_min = 0.5 * np.arctan2(theta_num, theta_den)
    return theta_max_min

def find_ellipse_parameters(Rmin, f, g, h, s):
    '''
    Finds the semi-major axis, a, semi-minor axis, b, the tilt and 
    the inclination of the smallest ellipse stretched by, f, that 
    is centred at (dx,dy) w.r.t. the centre of the eclipse with 
    duration te.

    Parameters
    ----------
    Rmin : array_like (2-D [:,None])
        array containing all the radii for a face-on disk with
        impact parameter dy { Rmin = np.hypot(te/2.,dy) }
    f : float, array_like (2-D)
        the y stretch factor of the smallest disk.
    g : float, array_like (2-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(Rmin**2 * f**2 - dy**2)) }
    h : float, array_like (2-D)
        array containing all the R growth factors neccesary to
        compensate for a stretch factor f < 1
        { h = np.hypot(te / 2., dy / f) / Rmin } 
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }

    Returns
    -------
    disk_radii : array_like (2-D)
        Array containing all the deprojected disk radii of the ellipses
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
    x1, y1 = shear_ellipse_point(Rmin, f, g, h, s, theta1)
    x2, y2 = shear_ellipse_point(Rmin, f, g, h, s, theta2)
    # find the semi-major and semi-minor axes
    R1 = np.hypot(x1,y1)
    R2 = np.hypot(x2,y2)
    disk_radii = np.maximum(R1,R2)
    proj_radii = np.minimum(R1,R2)
    # determine the inclination
    inclination = np.arccos(proj_radii / disk_radii)
    # determine the tilt
    tilt = np.arctan2(y1,x1) # assuming R1 > R2
    tilt_mask = R2 > R1 # sfind where above is not true
    tilt = tilt + tilt_mask*np.pi/2 # at above locations add np.pi/2
    return disk_radii, np.rad2deg(tilt), np.rad2deg(inclination)

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
    dx : array_like (2-D [None,:])
        array containing all the x-shifts of the disk centre
    dy : array_like (2-D [:,None])
        array containing all the impact parameters of the disks
        investigated.
    f : float, array_like (2-D)
        the y stretch factor of the smallest disk.
    g : float, array_like (2-D)
        array containing all the squeeze factors necessary to
        compensate for the stretch factor f 
        { g = te * f / (2 * np.sqrt(Rmin**2 * f**2 - dy**2)) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }
    
    Returns
    -------
    slope1 : array_like (2-D)
        Array containing all the slopes of the tangents of the left
        hand side. i.e. at (-te/2, -dy).
    slope2 : array_like (2-D)
        Array containing all the slopes of the tangents of the right
        hand side. i.e. at ( te/2, -dy).
    '''
    # determining coordinates on circle
    y = -dy
    x1 = -te/2. - dx
    x2 =  te/2. - dx
    # slopes are based on:
    # Rp**2 = xp**2 + yp**2 = (g*x - s*f*y)**2 + (f*y)**2
    # dy/dx = (s*f*g*y - g^2*x) / ((s**2 + 1)f**2*y - s*f*g*x)
    # left side
    dy1 = -s * f**2 * y - f**2 * x1
    dx1 = (s**2 * f**2 + g**2) * y + s * f**2 * x1
    slope1 = dy1/dx1
    # right side
    dy2 = -s * f**2 * y - f**2 * x2
    dx2 = (s**2 * f**2 + g**2) * y + s * f**2 * x2
    slope2 = dy2/dx2
    return slope1, slope2

#####################################################################
##################### NUMERICAL SUB ROUTINES ########################
#####################################################################

def find_stretch(te, dx, dy, Rmin, s, Rhill, Rhill_tol=1e-6):
    '''
    This function finds the stretch factor for each of the (dx,dy)
    locations that would cause the radius to be Rhill ± Rtol. This
    function is numerical so not ideal in its current form a search
    for an analytical solution is underway

    Parameters
    ----------
    te : float
        duration of the eclipse.
    dx : array_like (2-D [None,:])
        array containing all the x-shifts of the disk centre
    dy : array_like (2-D [:,None])
        array containing all the impact parameters of the disks
        investigated.
    Rmin : array_like (2-D [:,None])
        array containing all the radii for a face-on disk with
        impact parameter dy { Rmin = np.hypot(te/2.,dy) }
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }
    Rhill : float
        the hill radius of the system being investigated, i.e. the
        largest acceptable disk radius to investigate
    Rhill_tol : float
        given the numerical nature of this function a tolerance on
        Rhill should be provided, default is 1e-6 days
    
    Returns
    -------
    f_min : array_like (2-D)
        array of the smallest stretch factor to give a point (dx,dy)
        a radius of Rhill. Note that all points with f = 1 R > Rhill
        are excluded.
    f_max : array_like (2-D)
        array of the largest stretch factor to give a point (dx,dy)
        a radius of Rhill. Note that all points with f = 1 R > Rhill
        are excluded.
    '''
    # set up the process for increasing and decreasing f
    f_max_step = [ 2 * Rhill / te, 0]
    for k in range(2):
        # set up minimum radius boundary
        f = 1
        g = 1
        h = 1
        r, _, _ = find_ellipse_parameters(Rmin, f, g, h, s)
        # while loop - step size
        f_step = (f_max_step[k] - f) / 2.
        # while loop - bisection jump forward, backward or done
        steps = np.zeros_like(s)
        steps[r<Rhill] = 1
        # while loop - filter out grid points that are done
        use = (steps!=0)
        # while loop
        cond = np.sum(np.abs(steps))
        counter = 0
        while cond != 0:
            # set up f and g and h
            f += f_step * steps
            if k == 0:
                g = te * f / (2 * np.sqrt(Rmin**2 * f**2 - dy**2))
                h = 1
            elif k == 1:
                g = 1
                h = np.hypot(te/2., dy/f) / Rmin
            # determine a
            r, _, _ = find_ellipse_parameters(Rmin, f, g, h, s)
            # update steps
            steps[r>Rhill] = -1 # f too large
            steps[r<Rhill] =  1 # f too small
            steps[(r>=Rhill-Rhill_tol)*(r<=Rhill+Rhill_tol)] = 0 # r = Rhill±Rhill_tol
            steps *= use # ignore values where f already found
            # update use
            use = (steps!=0)
            # while loop
            cond = np.sum(np.abs(steps))
            counter += 1
            f_step /= 2.
            print('Trial %02i - # of grid points remaining %i'%(counter,cond))
        if k == 0:
            f_max = np.copy(f)
        else:
            f_min = np.copy(f)
    return f_min, f_max

def parameters_vs_stretch(te, dx, dy, Rmin, f_min, f_max, s, nf=20):
    '''
    This function determines how the various disk parameters vary with f.
    It creates 3-D data cubes for a, b, tilt, inclination, left and right
    gradients. Along the first 2 dimensions are the (dx,dy) grid positions
    along the third dimension are the same parameters with a different f
    value.
    
    Parameters
    ----------
    te : float
        duration of the eclipse.
    dx : array_like (2-D [None,:])
        array containing all the x-shifts of the disk centre
    dy : array_like (2-D [:,None])
        array containing all the impact parameters of the disks
        investigated.
    Rmin : array_like (2-D [:,None])
        array containing all the radii for a face-on disk with
        impact parameter dy { Rmin = np.hypot(te/2.,dy) }
    f_min : array_like (2-D)
        array of the smallest stretch factor to give a point (dx,dy)
        a radius of Rhill.
    f_max : array_like (2-D)
        array of the largest stretch factor to give a point (dx,dy)
        a radius of Rhill. 
    s : array-like (2-D)
        array containing all the shear factors for the different
        ellipses investigated { s = -dx / dy }
    nf : integer
        number of points from f0 to f (max) to investigate the 
        various ellipse parameters. default = 20

    Returns
    -------
    R : array_like (3-D)
        array containing the semi-major axes of the ellipses for
        each value of f
    T : array_like (3-D)
        array containing the tilt of the ellipses for each value 
        of f
    I : array_like (3-D)
        array containing the inclination of the ellipses for each
        value of f
    F : array_like (3-D)
        array containing the stretch factor f value at each step
        along the cube
    '''
    # define step sizeintial and final conditions
    f_step = (f_max - f_min) / (nf - 1)
    R, T, I, GL, GR, F = np.zeros((6,)+s.shape+(nf,))
    for x in range(nf):
        # define f, g and h
        fn = f_min + x * f_step
        gn = te * fn / (2 * np.sqrt(Rmin**2 * fn**2 - dy**2))
        hn = np.hypot(te/2., dy/fn) / Rmin
        # define masks (h = 1 if f > 1, g = 1 if f < 1)
        f_g  = fn <= 1 # define where g = 1
        f_h  = fn >= 1 # define where h = 1
        # apply masks
        gn[f_g] = 1
        hn[f_h] = 1
        # determine data points
        R[:,:,x], T[:,:,x], I[:,:,x] = find_ellipse_parameters(Rmin, fn, gn, hn, s)
        sl, sr = find_ellipse_slopes(te, dx, dy, fn, gn, s)
        GL[:,:,x] = np.abs(np.sin(np.arctan2(sl,1)))
        GR[:,:,x] = np.abs(np.sin(np.arctan2(sr,1)))
        F[:,:,x] = fn
    return R, T, I, GL, GR, F
    
def find_configuration(inc, tilt, Rhill, te, xmax, ymax, nx=50, ny=50, ymin=1e-8, inc_tol=0.1, tilt_tol=0.1, Rhill_tol=1e-6, nf=100):
    '''
    This function is used to solve for te, dx, dy, f to give a particular
    inclination and tilt. This is very involved, and may not even be possible
    so solutions are found numerically

    Parameters
    ----------
    inc : float
        the inclination angle of the disk
    tilt : float
        the tilt angle of the disk (ellipse w.r.t. x-axis)
    Rhill : float
        the hill radius of the system being investigated, i.e. the
        largest acceptable disk radius to investigate
    te : float
        duration of the eclipse.
    xmax : float
        maximum offset in the x direction of the centre of the 
        ellipse in days.
    ymax : float
        maximum offset in the y direction of the centre of the
        ellipse in days.
    nx : integer
        number of steps to investigate in the x direction. i.e.
        investigated space is np.linspace(-xmax, xmax, 2*nx).
    ny : integer
        number of steps to investigate in the y direction. i.e.
        investigated space is np.linspace(-ymax, ymax, 2*ny).
    ymin : float
        this is necessary because the shear parameter s = -dx/dy
        so dy != 0
    inc_tol : float
        the tolerance on the inclination angle found [default = 0.1].
    tilt_tol : float
        the tolerance on the tilt angle found [default = 0.1].
    Rhill_tol : float
        given the numerical nature of this function a tolerance on
        Rhill should be provided, default is 1e-6 days
    nf : integer
        number of f points calculated between f_min and f_max to find
        inclination and tilt

    Returns
    -------
    te : float
        duration of the eclipse.
    dx : float
        x - position of the centre of the ellipse
    dy : float
        y - position of the centre of the ellipse
    f : float
        stretch factor of the ellipse (from 0 to 2 * Rhill / te)
    '''
    # creating phase space
    dy = np.linspace(0, ymax, ny)
    dy[0] = ymin
    dx = np.linspace(0, xmax, nx)
    # reshape
    dy = dy[:,None]
    dx = dx[None,:]
    # important details
    Rmin = np.hypot(te/2., dy)
    s = -dx / dy
    # finding the stretch factor limits
    f_min, f_max = find_stretch(te, dx, dy, Rmin, s, Rhill, Rhill_tol)
    # finding the dependence on parameters with f
    R, T, I, GL, GR = parameters_vs_stretch(te, dx, dy, Rmin, f_min, f_max, nf)
    tilt_mask = (T >= tilt - tilt_tol) * (T <= tilt + tilt_tol)
    inc_mask  = (I >= inc - inc_tol) * (I <= inc + inc_tol)
    config_mask = tilt_mask * inc_mask
    xi, yi, fi = np.argwhere(config_mask).T


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

def investigate_ellipses(te, xmax, ymax, f, nx=50, ny=50, ymin=1e-8):
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

    Returns
    -------
    radii : array_like (2-D)
        Array containing the semi-major axes of the investigated
        ellipses.
    tilt : array_like (2-D)
        Array containing the tilt angles [deg] of the investigated
        ellipses. Tilt is the angle of the semi-major axis w.r.t.
        the x-axis
    inclination : array_like (2-D)
        Array containing the inclination angles [deg] of the
        investigated ellipses. Inclination is the angle obtained
        from the ratio of the semi-minor to semi-major axis
    grad_left : array_like (2-D)
        Array containing all the slopes of the left ellipse edge at
        the eclipse height. Note that this slope is defined as the
        absolute value of the sine of the arctangent of dy/dx.
    grad_right
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
    # reshape
    dy = dy[:,None]
    dx = dx[None,:]
    # important details
    Rmin = np.hypot(te/2., dy)
    s = -dx / dy
    # defining the squeeze factor g and growth factor h (related to f)
    if f < 1:
        h = np.hypot(te/2., dy/f) / Rmin
        g = 1
    elif f == 1:
        h = 1
        g = 1
    elif f > 1:
        g = (te * f) / (2 * np.sqrt(Rmin**2 * f**2 - dy**2))
    print(f,g,h)
    # investigating phase space
    r, tilt, inclination = find_ellipse_parameters(Rmin, f, g, h, s)
    slope_left, slope_right = find_ellipse_slopes(te, dx, dy, f, g, s)
    # filling the quadrants
    radii = reflect_quadrants(r)
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
######################## PLOTTING ROUTINES ##########################
#####################################################################

def plot_property(disk_prop, prop_name, te, xmax, ymax, f0, lvls="n", vmin=0, vmax=1e8, root=''):
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
    f0 : float
        the y stretch factor of the smallest disk.
    lvls : str, float, array_like (1-D) [default = "n"]
        either nothing specified (str), number of levels (float)
    vmin : float [default = 0]
        minimum value in the colourmap.
    vmax : float [default = 1e8]
        maximum value in the colourmap.
    root : string [default = '']
        string containing the path where the figure will be saved.

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
    plt.title('%s of Disk with $f$ = %.3f and $t_e$ = %.3f'%(title, f0, te))
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
    fig.savefig(root+'te_%.3f_f_%.3f_%s.png'%(te, f0, save))
    return None

def plot_stretch_vs_property(DISK_PROP, prop_name, te, dx, dy, F, configs=10, root=''):
    '''
    Plots the effect of the stretch factor on a geometrical property of the investigated ring system.
    
    Parameters
    ----------
    DISK_PROP : array_like (3-D)
        array containing a disk property for all the investigated
        ellipses and how it varies with f from f_min to f_max
        allowed by Rhill from function .
    prop_name : string
        string containing the name of the property
    te : float
        duration of the eclipse.
    dx : array_like (2-D [None,:])
        array containing all the x-shifts of the disk centre
    dy : array_like (2-D [:,None])
        array containing all the impact parameters of the disks
        investigated.
    F : array_like (3-D)
        array containing the stretch factor f value at each step
        along the cube of DISK_PROP
    configs: integer or array_like (2-D)
        configs is used to determine which configurations of the
        phase space are investigated. If configs is an integer
        that many random configurations will be plotted. If configs
        is an array it should have a (2,n) shape, (0,n) = dx INDICES
        and (1,n) = dy INDICES
    root : string [default = '']
        string containing the path where the figure will be saved. 
    
    Returns
    -------
    fig : root + "te_%.3f_nc_%i_%s_stretch.png" % (te, {configs or len(configs)}, prop_name)
    '''
    title = prop_name.capitalize()
    save = prop_name.lower()
    # determine what to plot
    if isinstance(configs,np.int):
        xi = np.random.randint(0,len(dx[0,:]),configs)
        yi = np.random.randint(0,len(dy[:,0]),configs)
    else:
        xi = configs[0]
        yi = configs[1]
    nc = len(xi)
    # create figure
    fig = plt.figure(figsize=(11,9))
    plt.xlabel('stretch factor, $f$')
    plt.ylabel(save)
    plt.title('Effect of $f$ on the %s of a Disk with $t_e$ of %.3f'%(title,te))
    for l in range(len(xi)):
        x = xi[l]
        y = yi[l]
        lbl = '($dx$,$dy$) = (%.3f,%.3f)'%(dx[0,x], dy[y,0])
        plt.plot(F[y,x],DISK_PROP[x,y],marker = 'o',label=lbl) 
    plt.legend(loc='best')
    fig.savefig(root+'te_%.3f_nc_%i_%s_stretch.png'%(te, nc, save))
    return None
