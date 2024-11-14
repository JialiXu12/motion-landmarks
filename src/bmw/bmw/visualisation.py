from mayavi import mlab
import automesh
import scipy
import bmw


def add_fig(viewer, label=''):
    # returns empty array if offscreen
    if viewer is not None:
        fig = viewer.Figure(label)
    else:
        fig = None
    return fig

def visualise_mesh(
        mesh, fig, visualise=False, face_colours=(1,0,0), pt_size=5, label=None,
        text=False, element_ids=False, text_elements=None, opacity=0.5,
        line_opacity = 1.,line_size = 1., elements=None, nodes=None, node_text=False, node_size=5,
        node_colours=(1,0,0), elements_to_display_nodes=None):
    if fig is not None:
        if label is None:
            label = mesh.label
        Xnid = mesh.get_node_ids(group='_default')
        
        if nodes=='all':
          nodes = Xnid[1] #[:-5]

        if visualise:
            # View breast surface mesh
            Xs, Ts = mesh.get_surfaces(res=16, elements=elements)
            if Xs.shape[0] == 0:
                Xs, Ts = mesh.get_faces(res=16, elements=elements)
            if elements is None:
                Xl = mesh.get_lines(res=32, internal_lines=False)
            else:
                Xl = mesh.get_lines(res=32, elements=elements, internal_lines=False)
            #import ipdb; ipdb.set_trace()
            fig.plot_surfaces('{0}_Faces'.format(label), Xs, Ts, color=face_colours,
                              opacity=opacity)
            #fig.plot_points('{0}_Nodes'.format(label), Xn, color=(1,0,1), size=pt_size)
            fig.plot_lines('{0}_Lines'.format(label), Xl, color=(1,1,0), size=line_size, opacity=line_opacity)
            if text_elements is not None:
                if text:
                    fig.plot_text('{0}_Text'.format(label), Xnid[0], Xnid[1], size=3)
            if elements_to_display_nodes is not None:
                for element_id in elements_to_display_nodes:
                    element = mesh.elements[element_id]
                    #import ipdb; ipdb.set_trace()
                    eXnid = mesh.get_node_ids(element.node_ids)
                    fig.plot_text('{0}_text_element{1}'.format(label, element.id), eXnid[0], eXnid[1], size=pt_size)
                    fig.plot_points('{0}_Nodes_element{1}'.format(label, element.id), eXnid[0], color=(1,0,1), size=pt_size/2)
            if element_ids:
                plot_element_ids(fig,'{0}_Xecid'.format(label), mesh, size=1, color=(0,0,0))
            if nodes is not None:
                #import ipdb; ipdb.set_trace()
                fig.plot_points(
                    '{0}_Points'.format(label), mesh.get_nodes(nodes), color=node_colours, size=node_size)
                if node_text:
                    fig.plot_text(
                        '{0}_Text'.format(label), mesh.get_nodes(nodes), nodes, size=3)


def plot_points(fig, fig_label_postfix, data, data_labels, visualise=True, point_size=10, text_size=5, colours=(1,0,0), plot_text=False):
    if fig is not None:
        if visualise:
            fig.plot_points(
                '{0}_Points'.format(fig_label_postfix), data, color=colours, size=point_size)
            if plot_text:
                fig.plot_text(
                    '{0}_Text'.format(fig_label_postfix), data, data_labels, size=text_size)


def plot_edges(fig, label, X, color=(1,0,0), opacity=1., line_width=1):
    from mayavi import mlab
    #mlab.figure(self.figure)
    #self.plot_points('landmarks', X, color=color, size=None, mode=None, opacity=opacity)
    src =  mlab.pipeline.scalar_scatter(X[:,0], X[:,1], X[:,2], opacity=0.,figure=fig.figure)
    delaunay =  mlab.pipeline.delaunay3d(src)
    delaunay.filter.offset = 999    # seems more reliable than the default
    edges = mlab.pipeline.extract_edges(delaunay)
    #import ipdb; ipdb.set_trace()
    mlab_obj = fig.plots.get(label)
    if mlab_obj != None:
        mlab_obj.name = 'Surface'
        fig.clear(label)

    fig.figure.scene.disable_render = True
    view = mlab.view()
    roll = mlab.roll()
    fig.plots[label] = mlab.pipeline.surface(edges, opacity=opacity, line_width=line_width, color=color,figure=fig.figure) #, opacity=opacity,  color=color, figure=self.figure)
    mlab.view(*view)
    mlab.roll(roll)
    fig.figure.scene.disable_render = False
    mlab_obj = fig.plots.get(label)
    mlab_obj.name = 'Surface'


def view_mri(image_path, fig, scan =None, axes='z_axes', slice_index=None):

    if scan == None:
        image = automesh.Scan(image_path)
    else:
        image = scan.copy()

    image.set_origin([0, 0, 0])
    coor, x, y, z = bmw.generate_image_coordinates(image.shape,
                                               image.spacing)

    src = mlab.pipeline.scalar_field(x,
                                     y,
                                     z,
                                     image.values.astype(
                                         scipy.int16))
    mlab.figure(fig.figure)
    outline = mlab.pipeline.outline(src, figure=fig.figure)

    if slice_index is None:
        slice_index=int(0.5 * image.num_slices)

    prone_ipw = mlab.pipeline.image_plane_widget(
        outline,
        plane_orientation=axes,
        slice_index=slice_index,
        colormap='black-white')

def plot_element_ids(fig , label, mesh, size=1, color=(0,0,0)):
    Xecids = mesh.get_element_cids()
    for idx, element in enumerate(mesh.elements):
        xi_dimension = len(element.basis)
        Xp = []
        if xi_dimension == 2:
            Xp = element.evaluate([0.5, 0.5], deriv=None)
        if xi_dimension == 3:
            Xp = element.evaluate([0.5,0.5,1], deriv=None)
        fig.plot_text('{0}{1}'.format(
            label, element.id), [Xp], [element.id], size=size,color=color)