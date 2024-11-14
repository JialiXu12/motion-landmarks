"""
ICP.py
iterative closest point fitting of 2 point clouds
Originally written by Ju Zhang. 
Modified by Thiranja, Prasad, Babarenda Gamage
"""
import scipy
from scipy.spatial import cKDTree
from scipy.optimize import leastsq, fmin

#======================================================================#
def transformRigid3D( x, t ):
    """ applies a rigid transform to list of points x.
    T = (tx,ty,tz,rx,ry,rz)
    """
    X = scipy.vstack( (x.T, scipy.ones(x.shape[0]) ) )
    T = scipy.array([[1.0, 0.0, 0.0, t[0]],\
               [0.0, 1.0, 0.0, t[1]],\
               [0.0, 0.0, 1.0, t[2]],\
               [1.0, 1.0, 1.0, 1.0]])
                
    Rx = scipy.array( [[1.0, 0.0, 0.0],\
                 [0.0, scipy.cos(t[3]), -scipy.sin(t[3])],\
                 [0.0, scipy.sin(t[3]),  scipy.cos(t[3])]] )
                 
    Ry = scipy.array( [[scipy.cos(t[4]), 0.0, scipy.sin(t[4])],\
                 [0.0, 1.0, 0.0],\
                 [-scipy.sin(t[4]), 0.0, scipy.cos(t[4])]] )
                 
    Rz = scipy.array( [[scipy.cos(t[5]), -scipy.sin(t[5]), 0.0],\
                 [scipy.sin(t[5]), scipy.cos(t[5]), 0.0],\
                 [0.0, 0.0, 1.0]] )
    
    T[:3,:3] = scipy.dot( scipy.dot( Rx,Ry ),Rz )
    return scipy.dot( T, X )[:3,:].T

def transformScale3D( x, S ):
    """ applies scaling to a list of points x. S = (sx,sy,sz)
    """
    return scipy.multiply( x, S )

#======================================================================#
def fitDataRigidScaleEPDP( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    dataTree = cKDTree( data )
    X = scipy.array(X)
    
    def obj( t ):
        xR = transformRigid3D( X, t[:6] )
        xRS = transformScale3D( xR, scipy.ones(3)*t[6] )
        d = dataTree.query( list(xRS) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = scipy.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
    tOpt = leastsq( obj, t0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, tOpt[:6] )
    XOpt = transformScale3D( XOpt, tOpt[6:] )
    
    return tOpt, XOpt
    
def fitDataRigidScaleDPEP( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points x to list of points data by minimising
    least squares distance between each point in data and closest
    neighbour in X
    """
    X = scipy.array(X)

    def obj( t ):
        xR = transformRigid3D( X, t[:6] )
        xRS = transformScale3D( xR, scipy.ones(3)*t[6] )
        xTree = cKDTree( xRS )
        d = xTree.query( list(data) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = scipy.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
    tOpt = leastsq( obj, t0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, tOpt[:6] )
    XOpt = transformScale3D( XOpt, tOpt[6:] )
    
    return tOpt, XOpt

#======================================================================#
def fitDataRigidTranslation( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    dataTree = cKDTree( data )
    X = scipy.array(X)
    
    def obj( t ):
        xR = transformRigid3D( X, scipy.array([t[0],t[1],t[2],0.,0.,0.]))
        d = dataTree.query( list(xR) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = scipy.array([0.0,0.0,0.0])
    tOpt = leastsq( obj, t0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, scipy.array([tOpt[0],tOpt[1],tOpt[2],0.,0.,0.]))
    
    return tOpt, XOpt



#======================================================================#
def alignCorrespondingDataRigidRotation( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    X = scipy.array(X)
    
    def obj( r ):
        xR = transformRigid3D( X, scipy.array([0.,0.,0.,r[0],r[1],r[2]]))
        import ipdb; ipdb.set_trace()
        err = data-xR
        d=scipy.zeros((data.shape[0]))
        for idx in range(data.shape[0]):
            d[idx] = scipy.linalg.norm(err[idx,:])
        #~ print d.mean()
        return d*d
        
    r0 = scipy.array([0.0,0.0,0.0])
    rOpt = leastsq( obj, r0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, scipy.array([0.,0.,0.,rOpt[0],rOpt[1],rOpt[2]]))
    
    return rOpt, XOpt

#======================================================================#
def alignCorrespondingDataRigidRotationZ( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    X = scipy.array(X)
    
    def obj( r ):
        xR = transformRigid3D( X, scipy.array([0.,0.,0.,0.,0.,r[0]]))
        err = data-xR
        d=scipy.zeros((data.shape[0]))
        for idx in range(data.shape[0]):
            d[idx] = scipy.linalg.norm(err[idx,:])
        #~ print d.mean()
        #import ipdb; ipdb.set_trace()
        return d*d
        
    r0 = scipy.array([0.0])
    rOpt = leastsq( obj, r0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, scipy.array([0.,0.,0.,0.,0.,rOpt[0]]))
    
    return rOpt, XOpt
