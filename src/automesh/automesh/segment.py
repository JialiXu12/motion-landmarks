import numpy as np
import pickle
import vtk

class PolydataFromImageParams(object):

    def __init__(self):
        self.smoothImage = 1
        self.imgSmthSD = 2.0
        self.imgSmthRadius = 1.5
        self.isoValue = 200.0
        self.smoothIt = 100
        self.smoothFeatureEdge = 0
        self.deciRatio = 0.5
        self.deciPerserveTopology = 0
        self.clean = True
        self.cleanPointMerging = 1
        self.cleanTolerance = 0.0
        self.filterNormal = 1
        self.calcCurvature = 1

    def save(self, filename):
        f = open(filename + '.polyparams', 'w')
        pickle.dump(self, f)
        f.close()





def image2Triangle(imageArray, isoValue, deciRatio=0.5, smoothIt=200):


    return V, T, N


