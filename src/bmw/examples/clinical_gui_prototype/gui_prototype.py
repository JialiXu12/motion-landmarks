# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
import sys
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)
import numpy,scipy
import automesh
import bmw
from mayavi import mlab
from traits.api import HasTraits, Instance, on_trait_change, Button, Range, Enum, Bool, Str
from traitsui.api import View, Item, HSplit, VSplit, Group,VGroup, CheckListEditor
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor
#import viewer
import morphic
import nibabel as nii
from scipy.spatial import cKDTree


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = scipy.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = scipy.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*scipy.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = scipy.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*scipy.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def points_to_plane(p0,p1,p3,p):
    u = p1 - p0
    v = p3 - p0
    # vector normal to plane
    n = numpy.cross(u, v)
    n /= numpy.linalg.norm(n)

    p_ = p - p0
    dist_to_plane = numpy.dot(p_, n)
    p_normal = numpy.dot(p_, n) * n
    p_tangent = p_ - p_normal

    closest_point = p_tangent + p0
    coords = numpy.linalg.lstsq(numpy.column_stack((u, v)), p_tangent)[0]
    print (closest_point )
    print (dist_to_plane)

    return n*dist_to_plane
def plot_mesh(fig,mesh,label, color=(0,1,1), line_colour=(1,1,0),opacity=0.75):
    Xn = mesh.get_nodes(group='_default')
    Xnid = mesh.get_node_ids(group='_default')
    Xf, Tf = mesh.get_surfaces(res=10)
    if Xf.shape[0] == 0:
        Xf, Tf = mesh.get_faces(res=10)
    Xl = mesh.get_lines(res=10)
    fig.plot_surfaces('Faces_{0}'.format(label), Xf, Tf, color=color, opacity=opacity)
    #fig.plot_points('Nodes', Xn, color=(1,0,1), size=2)
    #fig.plot_lines('Lines_{0}'.format(label), Xl, color=line_colour, size=5,opacity=0.1)
    # Takes a long time to render text
    #fig.plot_text('Text', Xnid[0], Xnid[1], size=2)

def plot_meshes(fig,labels,meshes,color):
    for mesh_idx, mesh in enumerate(meshes):
        plot_mesh(fig, mesh, labels[mesh_idx], color=color, line_colour=(0,1,0), opacity=0.2)

def generate_image_coordinates(image_shape, spacing):
    x, y, z = scipy.mgrid[0:image_shape[0],0:image_shape[1],0:image_shape[2]]
    x = x*spacing[0]
    y = y*spacing[1]
    z = z*spacing[2]
    image_coor = scipy.vstack((x.ravel(),y.ravel(),z.ravel())).transpose()
    return image_coor, x, y, z

class Figure:
    
    def __init__(self, scene=None):#, bgcolor=(.5,.5,.5),size=(400, 400)):
        self.figure = scene.mayavi_scene
        self.plots = {}
        
    def clear(self, label=None):
        if label == None:
            labels = self.plots.keys()
        else:
            labels = [label]
        mlab.figure(self.figure)
        
        for label in labels:
            mlab_obj = self.plots.get(label)
            if mlab_obj != None:
                if mlab_obj.name == 'Surface':
                    mlab_obj.parent.parent.parent.remove()
                else:
                    mlab_obj.parent.parent.remove()
                self.plots.pop(label)

    def get_camera(self):
        return (mlab.view(), mlab.roll())

    def set_camera(self, camera):
        mlab.view(*camera[0])
        mlab.roll(camera[1])

    def hide(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = False

    def show(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = True

    def plot_surfaces(self, label, X, T, scalars=None, color=None, rep='surface', opacity=1.0):
        #mlab.figure(self.figure)
        
        if color == None:
            color = (1,0,0)
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if scalars==None:
                self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color, opacity=opacity, representation=rep,figure=self.figure)
            else:
                self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars, opacity=opacity,figure=self.figure)
        
        else:
            self.figure.scene.disable_render = True
            view = mlab.view()
            roll = mlab.roll()
            
            if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                if scalars==None:
                    mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2],figure=self.figure)
                    mlab_obj.actor.property.color = color
                    mlab_obj.actor.property.opacity = opacity
                else:
                    mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=scalars, opacity=opacity,figure=self.figure)
                
                
            else:
                self.clear(label)
                if scalars==None:
                    self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, color=color, opacity=opacity, representation=rep,figure=self.figure)
                else:
                    self.plots[label] = mlab.triangular_mesh(X[:,0], X[:,1], X[:,2], T, scalars=scalars, opacity=opacity,figure=self.figure)
                
            mlab.view(*view)
            mlab.roll(roll)
            self.figure.scene.disable_render = False
            
    def plot_lines(self, label, X, color=None, size=0, opacity=1.):
        
        nPoints = 0
        for x in X:
            nPoints += x.shape[0]
        
        Xl = numpy.zeros((nPoints, 3))
        connections = []
        
        ind = 0
        for x in X:
            Xl[ind:ind+x.shape[0],:] = x
            for l in range(x.shape[0]-1):
                connections.append([ind + l, ind + l + 1])
            ind += x.shape[0]
        connections = numpy.array(connections)
        
       # mlab.figure(self.figure)
        
        if color == None:
            color = (1,0,0)
        if size == None:
            size = 1
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            self.plots[label] = mlab.points3d(Xl[:,0], Xl[:,1], Xl[:,2], color=color, scale_factor=0,figure=self.figure)
            self.plots[label].mlab_source.dataset.lines = connections
            mlab.pipeline.surface(self.plots[label], color=(1, 1, 1),
                              representation='wireframe',
                              line_width=size,
                              name='Connections', opacity=opacity,figure=self.figure)
        else:
            self.figure.scene.disable_render = True
            self.clear(label)
            self.plots[label] = mlab.points3d(Xl[:,0], Xl[:,1], Xl[:,2], color=color, scale_factor=0,figure=self.figure)
            self.plots[label].mlab_source.dataset.lines = connections
            #~ self.plots[label].mlab_source.update()
            mlab.pipeline.surface(self.plots[label], color=color,
                              representation='wireframe',
                              line_width=size,
                              name='Connections',figure=self.figure)
            self.figure.scene.disable_render = False
        
            
    def plot_lines2(self, label, X, scalars=None, color=None, size=0):
        
       # mlab.figure(self.figure)
        
        if color == None:
            color = (1,0,0)
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if scalars==None:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, tube_radius=size,figure=self.figure)
            else:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, tube_radius=size,figure=self.figure)
        
        else:
            self.figure.scene.disable_render = True
            #~ view = mlab.view()
            
            #~ if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                #~ if scalars==None:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2])
                    #~ mlab_obj.actor.property.color = color
                #~ else:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=scalars)
                #~ 
            #~ else:
                #~ self.clear(label)
                #~ if scalars==None:
                    #~ self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, line_width=size)
                #~ else:
                    #~ self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, line_width=size)
            
            self.clear(label)
            if scalars==None:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], color=color, tube_radius=size, reset_zoom=False,figure=self.figure)
            else:
                self.plots[label] = mlab.plot3d(X[:,0], X[:,1], X[:,2], scalars, tube_radius=size, reset_zoom=False,figure=self.figure)
            
            #~ mlab.view(*view)
            self.figure.scene.disable_render = False
            
    def plot_points(self, label, X, color=None, size=None, mode=None, opacity=1.):
        
        #mlab.figure(self.figure)
        
        if color==None:
            color=(1,0,0)
        
        if size == None and mode == None or size == 0:
            size = 1
            mode = 'point'
        if size == None:
            size = 1
        if mode==None:
            mode='sphere'
        
        if isinstance(X, list):
            X = numpy.array(X)
        
        if len(X.shape) == 1:
            X = numpy.array([X])
        
        mlab_obj = self.plots.get(label)
        if mlab_obj == None:
            if isinstance(color, tuple):
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode,opacity=opacity,figure=self.figure)
            else:
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none',opacity=opacity, mode=mode,figure=self.figure)
        
        else:
            #self.figure.scene.disable_render = True
            #view = mlab.view()
            #roll = mlab.roll()
            
            ### Commented out since VTK gives an error when using mlab_source.set
            #~ if X.shape[0] == mlab_obj.mlab_source.x.shape[0]:
                #~ if isinstance(color, tuple):
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2])
                    #~ mlab_obj.actor.property.color = color
                #~ else:
                    #~ mlab_obj.mlab_source.set(x=X[:,0], y=X[:,1], z=X[:,2], scalars=color)
                #~ 
                #~ 
            #~ else:
                #~ self.clear(label)
                #~ if isinstance(color, tuple):
                    #~ self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size, mode=mode)
                #~ else:
                    #~ self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size, scale_mode='none', mode=mode)
            
            self.clear(label)
            if isinstance(color, tuple):
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color=color, scale_factor=size,opacity=opacity, mode=mode,figure=self.figure)
            else:
                self.plots[label] = mlab.points3d(X[:,0], X[:,1], X[:,2], color, scale_factor=size,opacity=opacity, scale_mode='none', mode=mode,figure=self.figure)
                
            #mlab.view(*view)
            #mlab.roll(roll)
            #self.figure.scene.disable_render = False
            
            
    def plot_text(self, label, X, text, size=1, color=(1,1,1)):
        view = mlab.view()
        roll = mlab.roll()
        self.figure.scene.disable_rendeplot_element_idsr = True
        
        scale = (size, size, size)
        mlab_objs = self.plots.get(label)
        
        if mlab_objs != None:
            if len(mlab_objs) != len(text):
                for obj in mlab_objs:
                    obj.remove()
            self.plots.pop(label)
        
        mlab_objs = self.plots.get(label)
        if mlab_objs == None:
            text_objs = []
            for x, t in zip(X, text):
                text_objs.append(mlab.text3d(x[0], x[1], x[2], str(t), scale=scale, color=color))
            self.plots[label] = text_objs
        elif len(mlab_objs) == len(text):
            for i, obj in enumerate(mlab_objs):
                obj.position = X[i,:]
                obj.text = str(text[i])
                obj.scale = scale
        else:
            print ("HELP, I shouldn\'t be here!!!!!")
        
        mlab.view(*view)
        mlab.roll(roll)
        self.figure.scene.disable_render = False

    def plot_element_ids(self, label, mesh, size=1, color=(1,1,1)):
        Xecids = mesh.get_element_cids()
        for idx, element in enumerate(mesh.elements):
            Xp = element.evaluate([0.5,0.5], deriv=None)
            self.plot_text('{0}{1}'.format(
                label, element.id), [Xp], [element.id], size=5, color=color)


    def plot_edges(self, label, X, color=(1,0,0), opacity=1., line_width=1):

        #mlab.figure(self.figure)
        #self.plot_points('landmarks', X, color=color, size=None, mode=None, opacity=opacity)
        print (X)
        src =  mlab.pipeline.scalar_scatter(X[:,0], X[:,1], X[:,2], opacity=0.,figure=self.figure)
        delaunay =  mlab.pipeline.delaunay3d(src)
        delaunay.filter.offset = 999    # seems more reliable than the default
        edges = mlab.pipeline.extract_edges(delaunay)
        #import ipdb; ipdb.set_trace()
        mlab_obj = self.plots.get(label)
        if mlab_obj != None:
            mlab_obj.name = 'Surface'
            self.clear(label)

        self.figure.scene.disable_render = True
        view = mlab.view()
        roll = mlab.roll()
        self.plots[label] = mlab.pipeline.surface(edges, opacity=opacity, line_width=line_width, color=color,figure=self.figure) #, opacity=opacity,  color=color, figure=self.figure)
        mlab.view(*view)
        mlab.roll(roll)
        self.figure.scene.disable_render = False
        mlab_obj = self.plots.get(label)
        mlab_obj.name = 'Surface'
################################################################################
#The actual visualization
class Visualization(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    generate_landmarks_btn = Button('Generate landmarks')
    clear_labels_btn = Button('Clear labels')
    clear_landmarks_btn = Button('Clear landmarks')
    clear_points_btn = Button('Clear points')
    hide_show_points_or_landmarks_btn = Button('Hide/show points/landmarks')
    hide_show_model_btn = Button('Hide/show model')
    reset_views_btn = Button('Reset view')
    ptSize = Range(1, 15, 10, )#mode='spinner')
    stiffness = Range(1, 15, 10, )#mode='spinner')
    volunteer_dropdown = Enum(['CL00207','CL00211'])
       # ['CL00012','CL00027','CL00040','CL00062','CL00140','CL00169','CL00181','CL00192','CL00199',
       # 'CL00015','CL00029','CL00042','CL00068','CL00158','CL00174','CL00185','CL00193','CL00207',
       # 'CL00016','CL00035','CL00047','CL00138','CL00159','CL00176','CL00189','CL00196','CL00209',
       # 'CL00017','CL00038','CL00056','CL00139','CL00163','CL00179','CL00191','CL00198','CL00211'])
    label_checkbox = Bool('Select labels')
    x_plane_checkbox = Bool('x-plane')
    y_plane_checkbox = Bool('y-plane')
    z_plane_checkbox = Bool('z-plane')
    #visble_planes = List(CheckListEditor(values=['x-plane','y-plane','z-plane'],cols=3))
    selected_features = Enum("labels", "landmarks")


    prone_label = Str
    supine_label = Str

    @on_trait_change('reset_views_btn')
    def reset_views(self):
        if hasattr(self, 'prone_default_view_point1'):
            self.prone_ipw_z.ipw.point1 = self.prone_default_view_point1
            self.prone_ipw_z.ipw.point2 = self.prone_default_view_point2
            self.prone_ipw_z.ipw.origin = self.prone_default_view_origin
            self.prone_ipw_z.ipw.update_placement()

        if hasattr(self, 'supine_default_view_point1'):
            self.supine_ipw_z.ipw.point1 = self.supine_default_view_point1
            self.supine_ipw_z.ipw.point2 = self.supine_default_view_point2
            self.supine_ipw_z.ipw.origin = self.supine_default_view_origin
            self.supine_ipw_z.ipw.update_placement()

    @on_trait_change('volunteer_dropdown')
    def redraw_scene1(self):
        self.add_plots1(self.scene1)

    @on_trait_change('volunteer_dropdown')
    def redraw_scene2(self):
        self.add_plots2(self.scene2)

    def add_plots1(self,scene):
        mechanics_dir = '/home/psam012/Documents/opt/bmw/data/clinical_pipeline_results_2016_09_05/clinical_mechanics_meshes/'+self.volunteer_dropdown # os.path.join('/home/psam012/Documents/opt/bmw/data/2016-06-20/mechanics_meshes/',self.volunteer_dropdown)
        warping_dir = '/home/psam012/Documents/opt/bmw/data/clinical_pipeline_results_2016_09_05/clinical_warped_images/'+self.volunteer_dropdown
        mlab.clf(figure=scene.mayavi_scene)
        mlab.figure(scene.mayavi_scene)
        self.fig1 = Figure(scene=scene)

        nii_path = os.path.join(warping_dir,'prone_mri_aligned_with_model.nii.gz')
        nii_mask_path = os.path.join(warping_dir,'prone_mask.nii.gz')
        #import ipdb; ipdb.set_trace()
        if os.path.exists(nii_path):
            image= nii.load(nii_path)
            mask_image= nii.load(nii_mask_path)
            self.pixdim = image.header['pixdim'][1:4]
            self.img_shape = image.shape
            self.prone_coor, self.prone_x, self.prone_y, self.prone_z = generate_image_coordinates(self.img_shape,self.pixdim)
            src = mlab.pipeline.scalar_field(self.prone_x,self.prone_y,self.prone_z,image.get_data())
            outline = mlab.pipeline.outline(
                                        src,
                                        figure=scene.mayavi_scene,
                                        )

            self.prone_ipw_x = mlab.pipeline.image_plane_widget(
                                outline,
                                plane_orientation='x_axes',slice_index=int(0.5 * image.header.get_n_slices()),colormap='black-white')
            self.prone_ipw_y = mlab.pipeline.image_plane_widget(
                                outline,
                                plane_orientation='y_axes',slice_index=int(0.5 * image.header.get_n_slices()),colormap='black-white')
            self.prone_ipw_z = mlab.pipeline.image_plane_widget(
                                outline,
                                plane_orientation='z_axes',slice_index=int(0.5 * image.header.get_n_slices()),colormap='black-white')
            mask_src = mlab.pipeline.scalar_field(self.prone_x,self.prone_y,self.prone_z,mask_image.get_data())
            self.prone_mask_ipw = mlab.pipeline.image_plane_widget(
                                mask_src,
                                plane_orientation='z_axes',slice_index=int(0.5 * image.header.get_n_slices()),colormap='black-white',transparent=True,plane_opacity=0.5,opacity=0.5)
            self.prone_mask_ipw.parent.parent.scalar_data[:] = 0
            self.prone_default_view_point1 = self.prone_ipw_z.ipw.point1
            self.prone_default_view_point2 = self.prone_ipw_z.ipw.point2
            self.prone_default_view_origin = self.prone_ipw_z.ipw.origin
            self.prone_ipw_z.ipw.middle_button_action = 0


        prone_mesh_path = os.path.join(mechanics_dir,'prone.mesh')
        if os.path.exists(prone_mesh_path):
            self.prone_mesh = morphic.Mesh(prone_mesh_path)
            self.elem_points1, self.elem_xi1, self.elem_ids1 = bmw.mesh_generation.generate_points_in_elements(self.prone_mesh, num_points=30)
            self.tree1 = cKDTree(self.elem_points1)
            #import ipdb; ipdb.set_trace()
            plot_meshes(self.fig1, ['prone_mesh'], [self.prone_mesh], (0,1.,1))

        def select_point(obj, evt):
            view = mlab.view()
            roll = mlab.roll()
            position = obj.GetCurrentCursorPosition()
            point_coord = self.pixdim*position
            dd, point_idx = self.tree1.query(point_coord)
            embedded_pt_xi = self.elem_xi1[point_idx]
            embedded_pt_elem = self.elem_ids1[point_idx]
            transformed_pt = self.supine_mesh.elements[embedded_pt_elem].evaluate(embedded_pt_xi)

            p0 = self.prone_ipw_z.ipw.point1
            p1 = self.prone_ipw_z.ipw.point2
            p3 = self.prone_ipw_z.ipw.origin
            u = points_to_plane(p0,p1,p3,transformed_pt)#/self.pixdim)

            if self.label_checkbox:
                #import ipdb; ipdb.set_trace()
                self.fig1.plot_points('point', point_coord, color=(1.,0.,0.), size=self.ptSize,opacity=0.5)
                self.fig2.plot_points('point', transformed_pt, color=(1.,0.,0.), size=self.ptSize,opacity=0.5)
                self.supine_ipw_z.ipw.point1 = p0+u
                self.supine_ipw_z.ipw.point2 = p1+u
                self.supine_ipw_z.ipw.origin = p3+u
                self.supine_ipw_z.ipw.update_placement()
            else:
            #import ipdb; ipdb.set_trace()
                mask = sector_mask(self.img_shape[0:2],(position[0],position[1]),self.ptSize,(0,360))
                self.prone_mask_ipw.parent.parent.scalar_data[:,:,int(position[2])] = scipy.logical_or(mask,self.prone_mask_ipw.parent.parent.scalar_data[:,:,int(position[2])])

            self.prone_mask_ipw.ipw.point1 = p0
            self.prone_mask_ipw.ipw.point2 = p1
            self.prone_mask_ipw.ipw.origin = p3
            self.prone_mask_ipw.ipw.update_placement()

            mlab.view(*view)
            mlab.roll(roll)
            mlab.sync_camera(self.fig1.figure, self.fig2.figure)
        self.prone_ipw_z.ipw.add_observer('EndInteractionEvent', select_point)


    @on_trait_change('generate_landmarks_btn')
    def update_landmarks(self):
        idxs = self.prone_mask_ipw.parent.parent.scalar_data.ravel().nonzero()[0]
        self.fig1.plot_edges('landmark', self.prone_coor[idxs,:], color=(1,0,0), opacity=0.3)# line_width=5) 

        dd, point_idx = self.tree1.query(self.prone_coor[idxs,:])
        embedded_pts_xi = self.elem_xi1[point_idx]
        embedded_pts_elem = self.elem_ids1[point_idx]

        transformed_points = scipy.empty_like(embedded_pts_xi)
        for element_id in scipy.unique(embedded_pts_elem):
            element = self.supine_mesh.elements[element_id]
            unique_elem_idxs = scipy.where(embedded_pts_elem == element_id)[0]
            transformed_points[unique_elem_idxs,:] = element.evaluate(embedded_pts_xi[unique_elem_idxs,:])

        self.fig2.plot_edges('landmark', transformed_points, color=(1,0,0), opacity=0.3)# line_width=5) 

    @on_trait_change('clear_labels_btn')
    def clear_labels(self):
        self.prone_mask_ipw.parent.parent.scalar_data[:] = 0
        self.prone_mask_ipw.ipw.update_placement()

    @on_trait_change('clear_landmarks_btn')
    def clear_landmarks(self):
        self.fig1.clear('landmark')
        self.fig2.clear('landmark')

    @on_trait_change('clear_points_btn')
    def clear_points(self):
        self.fig1.clear('point')
        self.fig2.clear('point')

    @on_trait_change('hide_show_model_btn')
    def hide_model(self):
        #import ipdb; ipdb.set_trace()
        if self.fig1.plots['Faces_prone_mesh'].visible:
            self.fig1.plots['Faces_prone_mesh'].visible = False
            self.fig2.plots['Faces_supine_mesh'].visible = False
        else:
            self.fig1.plots['Faces_prone_mesh'].visible = True
            self.fig2.plots['Faces_supine_mesh'].visible = True
        #.seed.widget.enabled = False

    @on_trait_change('x_plane_checkbox')
    def hide_x_plane(self):
        if self.x_plane_checkbox:
            self.prone_ipw_x.visible = True
        else:
            self.prone_ipw_x.visible = False

    @on_trait_change('y_plane_checkbox')
    def hide_y_plane(self):
        if self.y_plane_checkbox:
            self.prone_ipw_y.visible = True
        else:
            self.prone_ipw_y.visible = False

    @on_trait_change('z_plane_checkbox')
    def hide_z_plane(self):
        if self.z_plane_checkbox:
            self.prone_ipw_z.visible = True
            self.prone_mask_ipw.visible = True
        else:
            self.prone_ipw_z.visible = False
            self.prone_mask_ipw.visible = False


    @on_trait_change('hide_show_points_or_landmarks_btn')
    def hide_points_or_landmarks(self):

        if self.label_checkbox:
            if self.fig1.plots['point'].visible:
                self.fig1.plots['point'].visible = False
                self.fig2.plots['point'].visible = False
            else:
                self.fig1.plots['point'].visible = True
                self.fig2.plots['point'].visible = True
        else:
            if 'landmark' in self.fig1.plots.keys():
                if self.fig1.plots['landmark'].visible:
                    self.fig1.plots['landmark'].visible = False
                    self.fig2.plots['landmark'].visible = False
                else:
                    self.fig1.plots['landmark'].visible = True
                    self.fig2.plots['landmark'].visible = True

    def add_plots2(self,scene):

        #mechanics_dir = './'+self.volunteer_dropdown#os.path.join('/home/psam012/Documents/opt/bmw/data/2016-06-20/mechanics_meshes/',self.volunteer_dropdown)
        mechanics_dir = '/home/psam012/Documents/opt/bmw/data/clinical_pipeline_results_2016_09_05/clinical_mechanics_meshes/'+self.volunteer_dropdown # os.path.join('/home/psam012/Documents/opt/bmw/data/2016-06-20/mechanics_meshes/',self.volunteer_dropdown)
        warping_dir = '/home/psam012/Documents/opt/bmw/data/clinical_pipeline_results_2016_09_05/clinical_warped_images/'+self.volunteer_dropdown
        parameter_set = 1
        # 207 has a tumour
        save_prefix = '_parameter_set_{0}'.format(parameter_set)

        mlab.clf(figure=scene.mayavi_scene)
        mlab.figure(scene.mayavi_scene)
        self.fig2 = Figure(scene=scene)

        nii_path = os.path.join(warping_dir,'warped_prone_mri_to_supine_aligned_with_model' + save_prefix + '_anterior.nii.gz')
        #import ipdb; ipdb.set_trace()
        if os.path.exists(nii_path):
            image= nii.load(nii_path)
            self.pixdim = image.header['pixdim'][1:4]
            self.img_shape = image.shape
            self.supine_coor, self.supine_x, self.supine_y, self.supine_z = generate_image_coordinates(self.img_shape,self.pixdim)
            src = mlab.pipeline.scalar_field(self.supine_x,self.supine_y,self.supine_z,image.get_data())
            outline = mlab.pipeline.outline(
                                        src,
                                        figure=scene.mayavi_scene,
                                        )
            self.supine_ipw_z = mlab.pipeline.image_plane_widget(
                                outline,
                                plane_orientation='z_axes',slice_index=int(0.5 * image.header.get_n_slices()),colormap='black-white')
            self.supine_default_view_point1 = self.supine_ipw_z.ipw.point1
            self.supine_default_view_point2 = self.supine_ipw_z.ipw.point2
            self.supine_default_view_origin = self.supine_ipw_z.ipw.origin
            self.supine_ipw_z.ipw.middle_button_action = 0

        supine_mesh_path = os.path.join(mechanics_dir,'supine' + save_prefix + '.mesh')
        if os.path.exists(supine_mesh_path):
            self.supine_mesh = morphic.Mesh(supine_mesh_path)
            self.elem_points2, self.elem_xi2, self.elem_ids2 = bmw.mesh_generation.generate_points_in_elements(self.supine_mesh, num_points=30)
            self.tree2 = cKDTree(self.elem_points2)
            plot_meshes(self.fig2, ['supine_mesh'], [self.supine_mesh], (0,0.5,1))

        def select_point(obj, evt):
            view = mlab.view()
            roll = mlab.roll()
            position = obj.GetCurrentCursorPosition()
            point_coord = self.pixdim*position
            dd, point_idx = self.tree2.query(point_coord)
            embedded_pt_xi = self.elem_xi2[point_idx]
            embedded_pt_elem = self.elem_ids2[point_idx]
            transformed_pt = self.prone_mesh.elements[embedded_pt_elem].evaluate(embedded_pt_xi)
            self.fig2.plot_points('point', point_coord, color=(1.,0.,0.), size=self.ptSize,opacity=0.5)
            self.fig1.plot_points('point', transformed_pt, color=(1.,0.,0.), size=self.ptSize,opacity=0.5)

            p0 = self.supine_ipw_z.ipw.point1
            p1 = self.supine_ipw_z.ipw.point2
            p3 = self.supine_ipw_z.ipw.origin
            u = points_to_plane(p0,p1,p3,transformed_pt)#/self.pixdim)

            self.prone_ipw_z.ipw.point1 = p0+u
            self.prone_ipw_z.ipw.point2 = p1+u
            self.prone_ipw_z.ipw.origin = p3+u
            self.prone_ipw_z.ipw.update_placement()

            self.prone_mask_ipw.ipw.point1 = p0+u
            self.prone_mask_ipw.ipw.point2 = p1+u
            self.prone_mask_ipw.ipw.origin = p3+u
            self.prone_mask_ipw.ipw.update_placement()

            mlab.view(*view)
            mlab.roll(roll)
            mlab.sync_camera(self.fig2.figure, self.fig1.figure)
        self.supine_ipw_z.ipw.add_observer('EndInteractionEvent', select_point)






#        mlab.clf(figure=self.scene2.mayavi_scene)
#        x, y, z, s = numpy.random.random((4, 100))
#        mlab.points3d(x, y, z, s, figure=self.scene2.mayavi_scene)

        #x, y, z, s = numpy.random.random((4, 100))
        #mlab.points3d(x, y, z, s, figure=self.scene1.mayavi_scene)


#    @on_trait_change('scene.activated')
#    def update_plot(self):
#         This function is called when the view is opened. We don't
#         populate the scene when the view is not yet open, as some
#         VTK features require a GLContext.

#         We can do normal mlab calls on the embedded scene.
#        self.scene1.mlab.test_points3d()

    # The layout of the dialog created
    view = View(HSplit(
                  VSplit(
                  Group(
                      'volunteer_dropdown','reset_views_btn','_','clear_points_btn','clear_labels_btn','_', 'generate_landmarks_btn','clear_landmarks_btn','_','hide_show_model_btn','hide_show_points_or_landmarks_btn',
                      show_labels=False,
                  ),
                  Group(
                 Item('label_checkbox',
                      label='Select points'),
                 Item('ptSize',
                      label='Label size'),
                 Item('x_plane_checkbox',
                      label='x-plane'),
                 Item('y_plane_checkbox',
                      label='y-plane'),
                 Item('z_plane_checkbox',
                      label='z-plane'),
                  )),
                  VGroup(
                 Item('prone_label',
                      label='Prone', style='readonly', show_label=False),
                      Item('scene1',
                           editor=SceneEditor(), height=250,
                           width=300),
                      show_labels=False,
                  ),
                  VGroup(
                 Item('supine_label',
                      label='Supine', style='readonly', show_label=False),
                      Item('scene2',
                           editor=SceneEditor(), height=250,
                           width=300, show_label=False),
                      show_labels=False,
                  ),
                ),
                resizable=True,
                )

################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


if __name__ == "__main__":
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.
    app = QtGui.QApplication.instance()
    container = QtGui.QWidget()
    container.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
    # define a "complex" layout to test the behaviour
    layout = QtGui.QGridLayout(container)

#    # put some stuff around mayavi
#    label_list = []
#    for i in range(3):
#        for j in range(3):
#            if (i==1) and (j==1):continue
#            label = QtGui.QLabel(container)
#            label.setText("Your QWidget at (%d, %d)" % (i,j))
#            label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
#            layout.addWidget(label, i, j)
#            label_list.append(label)

    label = QtGui.QLabel(container)
    label.setText("")
    label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
    layout.addWidget(label, 0, 0)

    mayavi_widget = MayaviQWidget(container)

    layout.addWidget(mayavi_widget, 1, 1)
    container.show()
    window = QtGui.QMainWindow()
    window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec_()
