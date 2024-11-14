# Standard library imports
import os
import numpy as np
# Related third party imports
import bmw
import automesh
from morphic import utils
import morphic


class SubjectModel(object):

    def __init__(self,surface_mesh_path='',torso_mesh_path='',
                 image_path='',metadata=None,position=None,vl_id =None,
                 soft_landmarks = None):
        self.metadata = metadata
        self.cw_surface_mesh = {}
        self.lskin_surface_mesh = {}
        self.rskin_surface_mesh = {}
        self.torsoMesh = {}
        self.scan = {}
        self.position= position
        self.image_path = ''
        self.surface_mesh_path = ''
        self.mesh_path = ''
        self.vl_id = vl_id
        self.soft_landmarks = {}


        if vl_id is not None:
            self.position = position
            vl_id_str = 'VL{0:05d}'.format(vl_id)

            vl_image_path = os.path.join(image_path,vl_id_str,position)
            if os.path.exists(vl_image_path) and os.listdir(vl_image_path):
                self.image_path= vl_image_path
                self.scan = automesh.Scan(self.image_path)
                self.scan.origin = np.zeros(3)
            vl_mesh_path = torso_mesh_path
            if os.path.exists(vl_mesh_path):
                self.mesh_path = vl_mesh_path
                self.torsoMesh = morphic.Mesh(vl_mesh_path)



            if os.path.exists(surface_mesh_path):
                # Load mesh
                self.surface_mesh_path = os.path.join(surface_mesh_path, vl_id_str)
                try:
                    cw_mesh = bmw.load_chest_wall_surface_mesh(position)
                    self.cw_surface_mesh = utils.convert_hermite_lagrange(cw_mesh, tol=1e-9)
                except:
                    print ('ribcage mesh is empty')
                    pass

                try:
                    self.lskin_surface_mesh = bmw.load_skin_surface_mesh(position, 'left')
                # self.lskin_surface_mesh = self.convert_skin_hermite_lagrange(lskin,self.position)
                except:
                    print ('Left skin surface mesh is empty')
                    pass
                try:
                    self.rskin_surface_mesh = bmw.load_skin_surface_mesh(position, 'right')
                # self.rskin_surface_mesh = self.convert_skin_hermite_lagrange(rskin,self.position)
                except:
                    print ('Right skin surface mesh is empty')
                    pass


    def set_cw_surface(self,lagrange_mesh):
        self.cw_surface_mesh = lagrange_mesh

    def set_lskin_surface(self,lagrange_mesh):
        self.lskin_surface_mesh = lagrange_mesh

    def set_rskin_surface(self,lagrange_mesh):
        self.rskin_surface_mesh = lagrange_mesh
    def set_metadata(self,value):
        self.metadata = value

    def get_scan(self):

        return self.scan

    def load_chest_wall_surface_mesh(self, base_mesh):
        # Load fitted chest wall surface (cwm)
        cwm_fname = os.path.join(self.surface_mesh_path, 'ribcage_{0}.mesh'.format(base_mesh))
        if os.path.exists(cwm_fname):
            cwm = morphic.Mesh(cwm_fname)
            cwm.label = base_mesh + '_ribcage'
            return cwm
        else:
            raise ValueError('ribcage mesh not found')

    def load_lung_surface_mesh(self, base_mesh):
        # Load fitted lung surface mesh (lung)
        lm_fname = os.path.join(self.surface_mesh_path, 'lungs_{0}.mesh'.format(base_mesh))
        if os.path.exists(lm_fname):
            lm = morphic.Mesh(lm_fname)
            lm.label = base_mesh + '_lung'
            return lm
        else:
            raise ValueError('lung surface mesh not found')

    def load_skin_surface_mesh(self, base_mesh, side):
        # Load fitted skin surface mesh (skin)
        sm_fname = os.path.join(self.surface_mesh_path, 'skin_' + side + '_{0}.mesh'.format(base_mesh))
        if os.path.exists(sm_fname):
            sm = morphic.Mesh(sm_fname)
            sm.label = base_mesh + '_skin_' + side
            return sm
        else:
            raise ValueError('skin_' + side + ' surface mesh not found')

    def load_volume_mesh(self, base_mesh):
        # Load volume mesh
        fname = os.path.join(self.mesh_path, '{0}.mesh'.format(base_mesh))
        if os.path.exists(fname):
            m = morphic.Mesh(fname)
            m.label = base_mesh
            return m
        else:
            raise ValueError('{0}.mesh'.format(base_mesh) + ' volume mesh not found')
