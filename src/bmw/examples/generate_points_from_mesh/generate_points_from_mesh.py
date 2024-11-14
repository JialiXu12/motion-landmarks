import vtk
import scipy
import morphic
reload(morphic)

visualise = True
if visualise:
    from morphic import viewer
    if "fig" not in locals():
        fig = viewer.Figure()

# Load surface meshes
mesh = morphic.Mesh('./VL00046_volume.mesh')
Xn = mesh.get_nodes(group='_default')
Xnid = mesh.get_node_ids(group='_default')

if visualise:
    Xf, Tf = mesh.get_faces(res=10)
    Xl = mesh.get_lines(res=10)
    
    fig.plot_surfaces('Faces', Xf, Tf, color=(0,1,0), opacity=0.25)
    #fig.plot_points('Nodes', Xn, color=(1,0,1), size=2)
    fig.plot_lines('Lines', Xl, color=(1,1,0), size=5)
    # Takes a long time to render text
    #fig.plot_text('Text', Xnid[0], Xnid[1], size=2)

# Generate a grid of points within each element
num_Xe = len(mesh.elements.ids)
num_points_per_elem_xi = 5
if num_points_per_elem_xi > 100:
    raise ValueError('Warning: this will generate >1 million points')
total_num_points = num_Xe*num_points_per_elem_xi**3
xi = scipy.linspace(0., 1., num_points_per_elem_xi)
X, Y, Z = scipy.meshgrid(xi, xi, xi)
Xi3d = scipy.array([
    X.reshape((X.size)),
    Y.reshape((Y.size)),
    Z.reshape((Z.size))]).T
points = scipy.zeros((num_Xe, num_points_per_elem_xi**3, 3))
for Xeid, Xe in enumerate(mesh.elements):
    points[Xeid,:,:] = Xe.evaluate(Xi3d)
points = scipy.reshape(points, (total_num_points,3))

visualise = False
if visualise:
    if num_points_per_elem_xi < 5:
        fig.plot_points('Points', points, color=(1,1,0), size=1)

# Convert to vtk
vtk_points = vtk.vtkPoints()
for p in points:
    vtk_points.InsertNextPoint(p)

