# Circumplanetary_Rings

This Repository has two main goals

    1) Simulate Light Curves of a Transiting Circumplanetary Ring System across a Star
    2) Find the Possible Circumplanetary Ring System Solutions that could produce a given Light Curve
    
Simulating a Light Curve takes several input parameters, namely the centre of the ring system w.r.t. the centre of the eclipse (dx,dy), which is related to the the tilt, and inclination of the system, the ring structure (radii and transmission), the size of the star, and the resolution of the convolution grid. Outputs include the Light Curve, the Transmission Profile and the maximum Gradient of a Ring Sytem in this Particular Configuration.

Fitting a Light Curve is done by investigating a phase space of ring system centres (dx,dy), and size factor (related to the limiting hill sphere radius). Limits are imposed by the hill sphere and the measured gradients of the light curve. Outputs will be several arrays, and plots that show case the set of solutions for a particular Light Curve.

Notes
-----
This Repository is organised into two modules (routines_ring_fitting, and routines_ring_simulating) that contain all the routines and sub-routines necessary to accomplish [1] and [2]. These modules will also show (or prove) the working of the routines in each module.

Separate Scripts will be created with a command line option parser to use the modules for data processing
