import os
from opencmiss.iron import iron
import mesh_tools

if __name__ == '__main__':
    
    # Define mesh
    interpolation = iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE
    num_elements = [2, 2, 2]
    mesh_type = iron.GeneratedMeshTypes.REGULAR
    region, decomposition, mesh, geometric_field = \
        mesh_tools.generate_opencmiss_geometry(
            interpolation=interpolation,
            num_elements=num_elements,
            mesh_type=mesh_type)

    # Define data
    num_points_per_elem_xi = 4
    _, element_nums = mesh_tools.num_element_get(mesh, mesh_component=1)
    values, xi, elements = mesh_tools.interpolate_opencmiss_field_sample(
        geometric_field, element_ids=element_nums, 
        num_values=num_points_per_elem_xi, dimension=3, derivative_number=1,
        unique=True, geometric_field=geometric_field)
    if not os.path.exists('results'):
        os.makedirs('results')
    mesh_tools.export_datapoints_exdata(
        values, 'data_points', os.path.join('results/data_points'))

    # Define fitting parameters
    tau = 0
    kappa = 0
    smoothing_parameters = ([
        tau,  # tau_1
        kappa,  # kappa_11
        tau,  # tau_2
        kappa,  # kappa_22
        kappa,  # kappa_12
        tau,  # tau_3
        kappa,  # kappa_33
        kappa,  # kappa_13
        kappa])  # kappa_23

    fitting = mesh_tools.Fitting(fitting_type='data')
    fitting.set_results_folder('results/')
    fitting.set_region(region)
    fitting.set_mesh(mesh)
    fitting.set_decomposition(decomposition)
    fitting.set_geometric_field(geometric_field)
    fitting.set_data_positions(
        element_xi=xi, element_nums=elements, data_point_positions=values)
    fitting.set_data_values(values=values, labels=['a', 'b', 'c'])
    fitting.setup_fields()
    fitting.setup_equations()
    fitting.setup_problem()
    fitting.set_smoothing_parameters(smoothing_parameters)
    fitting.solve()
    fitting.export(iteration_num=1)
    fitting.finalise()

