"""
ICP.py
iterative closest point fitting of 2 point clouds
Originally written by Ju Zhang. 
Modified by Thiranja, Prasad, Babarenda Gamage
"""
import scipy
from scipy.spatial import KDTree
from scipy.optimize import leastsq, fmin
import numpy as np

#======================================================================#
def transformRigid3D( x, t ):
    """ applies a rigid transform to list of points x.
    T = (tx,ty,tz,rx,ry,rz)
    """
    x = np.array(x)
    X = scipy.vstack( (x.T, scipy.ones(x.shape[0]) ) )
    T = np.array([[1.0, 0.0, 0.0, t[0]],\
               [0.0, 1.0, 0.0, t[1]],\
               [0.0, 0.0, 1.0, t[2]],\
               [1.0, 1.0, 1.0, 1.0]])
                
    Rx = np.array( [[1.0, 0.0, 0.0],\
                 [0.0, scipy.cos(t[3]), -scipy.sin(t[3])],\
                 [0.0, scipy.sin(t[3]),  scipy.cos(t[3])]] )
                 
    Ry = np.array( [[scipy.cos(t[4]), 0.0, scipy.sin(t[4])],\
                 [0.0, 1.0, 0.0],\
                 [-scipy.sin(t[4]), 0.0, scipy.cos(t[4])]] )
                 
    Rz = np.array( [[scipy.cos(t[5]), -scipy.sin(t[5]), 0.0],\
                 [scipy.sin(t[5]), scipy.cos(t[5]), 0.0],\
                 [0.0, 0.0, 1.0]] )
    
    T[:3,:3] = scipy.dot( np.dot( Rx,Ry ),Rz )
    return scipy.dot( T, X )[:3,:].T


def transformTranslate3D(x, t):
    """ applies a rigid transform to list of points x.
    T = (tx,ty,tz,rx,ry,rz)
    """
    X = scipy.vstack((x.T, scipy.ones(x.shape[0])))
    T = np.array([[1.0, 0.0, 0.0, t[0]], \
                     [0.0, 1.0, 0.0, t[1]], \
                     [0.0, 0.0, 1.0, t[2]], \
                     [1.0, 1.0, 1.0, 1.0]])

    return scipy.dot(T, X)[:3, :].T

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
    
    dataTree = KDTree( data )
    X = np.array(X)
    
    def obj( t ):
        xR = transformRigid3D( X, t[:6] )
        xRS = transformScale3D( xR, scipy.ones(3)*t[6] )
        d = dataTree.query( list(xRS) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
    tOpt = leastsq( obj, t0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, tOpt[:6] )
    XOpt = transformScale3D( XOpt, tOpt[6:] )
    
    return tOpt, XOpt
    
def fitDataRigidScaleDPEP( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points x to list of points data by minimising
    least squares distance between each point in data and closest
    neighbour in X
    """
    X = np.array(X)

    def obj( t ):
        xR = transformRigid3D( X, t[:6] )
        xRS = transformScale3D( xR, scipy.ones(3)*t[6] )
        xTree = KDTree( xRS )
        d = xTree.query( list(data) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
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
    
    dataTree = KDTree( data )
    X = np.array(X)
    
    def obj( t ):
        xR = transformRigid3D( X, np.array([t[0],t[1],t[2],0.,0.,0.]))
        d = dataTree.query( list(xR) )[0]
        #~ print d.mean()
        return d*d
        
    t0 = np.array([0.0,0.0,0.0])
    tOpt = leastsq( obj, t0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, np.array([tOpt[0],tOpt[1],tOpt[2],0.,0.,0.]))
    
    return tOpt, XOpt



#======================================================================#
def alignCorrespondingDataRigidRotation( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    X = np.array(X)
    
    def obj( r ):
        xR = transformRigid3D( X, np.array([0.,0.,0.,r[0],r[1],r[2]]))
        err = data-xR
        d=scipy.zeros((data.shape[0]))
        for idx in range(data.shape[0]):
            d[idx] = scipy.linalg.norm(err[idx,:])
        #~ print d.mean()
        return d*d
        
    r0 = np.array([0.0,0.0,0.0])
    rOpt = leastsq( obj, r0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, np.array([0.,0.,0.,rOpt[0],rOpt[1],rOpt[2]]))
    
    return rOpt, XOpt


#======================================================================#
def alignCorrespondingDataRigidRotationTranslation( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    X = np.array(X)
    
    def obj( r ):
        xR = transformRigid3D( X, np.array(r))
        #import ipdb; ipdb.set_trace()
        err = data-xR
        d=scipy.zeros((data.shape[0]))
        for idx in range(data.shape[0]):
            d[idx] = scipy.linalg.norm(err[idx,:])
        print (d.mean())
        return d*d
        
    r0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
    rOpt = leastsq( obj, r0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, np.array(rOpt))
    
    return rOpt, XOpt


# ======================================================================#
def alignCorrespondingDataAndLandmarksRigidRotationTranslation(
        X, data,landmark_source,landmark_targets, xtol=1e-5, maxfev=0,
        weighting=100., rotation=True,
        r0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
    """ fit list of points x to list of points data by minimising
    least squares distance between each point in data and closest
    neighbour in X
    """
    X = np.array(X)

    def obj(t):
        xR = transformRigid3D(X, t)
        xTree = KDTree(xR)
        d = xTree.query(list(data))[0]
        # ~ print d.mean()

        landmark_sourceR = transformRigid3D(landmark_source, t)
        err = landmark_targets-landmark_sourceR
        dd = scipy.zeros((landmark_sourceR.shape[0]))
        for idx in range(landmark_sourceR.shape[0]):
            dd[idx] = scipy.linalg.norm(err[idx, :])

        return scipy.sum(d * d) + scipy.sum(dd * dd)*weighting

    def objTranslate(t):
        xR = transformTranslate3D(X, t)
        xTree = KDTree(xR)
        d = xTree.query(list(data))[0]
        # ~ print d.mean()

        landmark_sourceR = transformTranslate3D(landmark_source, t)
        err = landmark_targets-landmark_sourceR
        dd = scipy.zeros((landmark_sourceR.shape[0]))
        for idx in range(landmark_sourceR.shape[0]):
            dd[idx] = scipy.linalg.norm(err[idx, :])

        return np.sum(d * d) + np.sum(dd * dd)*weighting

    #res = scipy.optimize.minimize(obj_corres, r0, method='SLSQP',
                                 # options={'disp': True,'maxiter': 1000})
    res = scipy.optimize.minimize(objTranslate, r0, method='COBYLA',
                                  options={'disp': True,'maxiter': 100000})
    res = scipy.optimize.minimize(obj, res.x, method='COBYLA',
                                  options={'disp': True,'maxiter': 100000})
    print (res)
    return res

# ======================================================================#
def alignCorrespondingLandmarksRigidTransform(
        landmark_source,landmark_targets,z_comp_landmarks_source=[],z_com_landmarks_target = [], xtol=1e-5, maxfev=0,
        r0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
    """ fit list of points x to list of points data by minimising
    least squares distance between each point
    """
    def obj(t):

        landmark_sourceR = transformRigid3D(landmark_source, t)
        err = landmark_targets-landmark_sourceR
        dd = np.zeros((landmark_sourceR.shape[0]))
        for idx in range(landmark_sourceR.shape[0]):
            dd[idx] = scipy.linalg.norm(err[idx, :])

        return  np.sum(dd * dd)


    res = scipy.optimize.minimize(obj, r0, method='COBYLA',
                                  options={'disp': True,'maxiter': 100000})
    print (res)
    return res

def alignCorrespondingLandmarksRigidRotationTranslation (landmark_source,landmark_targets, xtol=1e-5, maxfev=0,
        r0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):

    def obj(t):

        landmark_sourceR = transformRigid3D(landmark_source, t)
        err = landmark_targets - landmark_sourceR
        dd = np.zeros((landmark_sourceR.shape[0]))
        for idx in range(landmark_sourceR.shape[0]):
            dd[idx] = scipy.linalg.norm(err[idx, :])

        return  np.sum(dd * dd)

    def objTranslate(t):

        landmark_sourceR = transformTranslate3D(landmark_source, t)
        err = landmark_targets - landmark_sourceR
        dd = np.zeros((landmark_sourceR.shape[0]))
        for idx in range(landmark_sourceR.shape[0]):
            dd[idx] = scipy.linalg.norm(err[idx, :])

        return np.sum(dd * dd)

    res = np.optimize.minimize(objTranslate, r0, method='COBYLA',
                                  options={'disp': True, 'maxiter': 100000})
    res = np.optimize.minimize(obj, res.x, method='COBYLA',
                                  options={'disp': True, 'maxiter': 100000})
    print (res)
    return res

#======================================================================#
def alignCorrespondingDataRigidRotationZ( X, data, xtol=1e-5, maxfev=0 ):
    """ fit list of points X to list of points data by minimising
    least squares distance between each point in X and closest neighbour
    in data
    """
    
    X = np.array(X)
    
    def obj( r ):
        xR = transformRigid3D( X, np.array([0.,0.,0.,0.,0.,r[0]]))
        err = data-xR
        d=np.zeros((data.shape[0]))
        for idx in range(data.shape[0]):
            d[idx] = scipy.linalg.norm(err[idx,:])
        #~ print d.mean()
        #import ipdb; ipdb.set_trace()
        return d*d
        
    r0 = np.array([0.0])
    rOpt = leastsq( obj, r0, xtol=xtol, maxfev=maxfev )[0]
    XOpt = transformRigid3D( X, np.array([0.,0.,0.,0.,0.,rOpt[0]]))
    
    return rOpt, XOpt


def computeLandmarkBasedError(landmark_sourceR, landmark_targets):
    err = landmark_targets - landmark_sourceR
    dd = np.zeros((landmark_sourceR.shape[0]))
    for idx in range(landmark_sourceR.shape[0]):
        dd[idx] = scipy.linalg.norm(err[idx, :])


    return np.mean(dd), np.std(dd)
