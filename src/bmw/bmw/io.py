import os
import morphic

def export_datapoints_exdata(data, label, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.  
    
    field_id = open(filename + '.exdata', 'w')
    field_id.write(' Group name: {0}\n'.format(label))
    field_id.write(' #Fields=1\n')
    field_id.write(' 1) coordinates, coordinate, rectangular cartesian, #Components=3\n')
    field_id.write('  x.  Value index=1, #Derivatives=0, #Versions=1\n')
    field_id.write('  y.  Value index=2, #Derivatives=0, #Versions=1\n')
    field_id.write('  z.  Value index=3, #Derivatives=0, #Versions=1\n')
    
    for point_idx, point in enumerate(range(1,data.shape[0]+1)):
        field_id.write(' Node: {0}\n'.format(point))
        for value_idx in range(data.shape[1]):
            field_id.write(' {0:.12E}\n'.format(data[point_idx,value_idx]))
    field_id.close()

def load_chest_wall_surface_mesh(mesh_path, base_mesh):
    # Load fitted chest wall surface (cwm)
    cwm_fname = os.path.join(mesh_path, 'ribcage_{0}.mesh'.format(base_mesh))
    if os.path.exists(cwm_fname):
        cwm = morphic.Mesh(cwm_fname)
        cwm.label = base_mesh+'_ribcage'
        return cwm
    else:
        raise ValueError('ribcage mesh not found')

def  load_lung_surface_mesh(mesh_path, base_mesh):
    # Load fitted lung surface mesh (lung)
    lm_fname = os.path.join(mesh_path, 'lungs_{0}.mesh'.format(base_mesh))
    if os.path.exists(lm_fname):
        lm = morphic.Mesh(lm_fname)
        lm.label = base_mesh+'_lung'
        return lm
    else:
        raise ValueError('lung surface mesh not found')

def load_skin_surface_mesh(mesh_path, base_mesh,side):
    # Load fitted skin surface mesh (skin)
    sm_fname = os.path.join(mesh_path, 'skin_'+side+'_{0}.mesh'.format(base_mesh))
    if os.path.exists(sm_fname):
        sm = morphic.Mesh(sm_fname)
        sm.label = base_mesh+'_skin_'+side
        return sm
    else:
        raise ValueError('skin_'+side + ' surface mesh not found')

def load_volume_mesh(mesh_path, base_mesh):
    # Load volume mesh
    fname = os.path.join(mesh_path, '{0}.mesh'.format(base_mesh))
    if os.path.exists(fname):
        m = morphic.Mesh(fname)
        m.label = base_mesh
        return m
    else:
        raise ValueError('{0}.mesh'.format(base_mesh) + ' volume mesh not found')