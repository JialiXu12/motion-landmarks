#!/usr/bin/env python

#> \file
#> \author Thiranja Prasad Babarenda Gamage
#> \brief 
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is Breast Modelling in OpenCMISS
#>
#> The Initial Developer of the Original Code is:
#> Thiranja Prasad Babarenda Gamage
#> Auckland, New Zealand, 
#> All Rights Reserved.
#>
#> Contributor(s):
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. If you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. If you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>

#> Main script

# Standard library imports
import shutil
import sys
import os
import pprint
import ipdb

# Local application/library specific imports
import h5py
import scipy

sys.path.insert(1, os.sep.join((os.environ['OPENCMISS_ROOT'],
                                       'cm', 'bindings','python')))
from opencmiss import CMISS
from bmw import mechanics
import morphic
import bmw
#import CMISS

def run():

    prone_path = 'volume.mesh'
    mesh3D = morphic.Mesh(prone_path)
    fig = None
    volunteer = 'VL00048'
    op = bmw.volunteer_setup(volunteer)
    op.src = None
    op.image_coor = None
    if not os.path.exists(op.results_dir):
        os.makedirs(op.results_dir)

    # Specify volume mesh element groups
    skin = False
    if skin:
        cranial_elem = range(11) + range(66,77)
        caudal_elem = range(55,66) + range(121,132)
        sternum_elem = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121]
        spine_elem = [10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 131]
        chestwall_elem = range(66)
        skin_elem = range(66,132)
    else:
        cranial_elem = range(11)
        caudal_elem = range(55,66)
        sternum_elem = [0,11,22,33,44,55]
        spine_elem = [10,21,32,43,54,65]
        chestwall_elem = range(66)
        skin_elem = range(66)

    for element in mesh3D.elements[cranial_elem]:
        element.add_to_group('cranial')
    for element in mesh3D.elements[caudal_elem]:
        element.add_to_group('caudal')
    for element in mesh3D.elements[sternum_elem]:
        element.add_to_group('sternum')
    for element in mesh3D.elements[spine_elem]:
        element.add_to_group('spine')
    for element in mesh3D.elements[chestwall_elem]:
        element.add_to_group('chestwall')
    for element in mesh3D.elements[skin_elem]:
        element.add_to_group('skin')

    #import ipdb; ipdb.set_trace()
    parameters = scipy.array([0.2, 5.])

    [converged, dependent_field1, decomposition, region] = mechanics.solve(
        fig, op, 0, parameters, mesh=mesh3D)
    if converged:
        offset = 1
        num_deriv = 1
        for node in mesh3D.nodes:
            for comp_idx in range(3):
                for deriv_idx in range(num_deriv):
                    node.values[comp_idx] = dependent_field1.ParameterSetGetNodeDP(
                        CMISS.FieldVariableTypes.U,
                        CMISS.FieldParameterSetTypes.VALUES,
                        1, deriv_idx + 1, node.id + offset, comp_idx + 1)
        mesh3D.save('{0}/reference.mesh'.format(op.results_dir))

        
        [converged, dependent_field2, decomposition, region] = mechanics.solve(
            fig, op, 1, parameters, mesh=mesh3D, previous_dependent_field=dependent_field1,
            decomposition=decomposition, region=region)
        if converged:
            for node in mesh3D.nodes:
                for comp_idx in range(3):
                    for deriv_idx in range(num_deriv):
                        node.values[comp_idx] = dependent_field2.ParameterSetGetNodeDP(
                            CMISS.FieldVariableTypes.U,
                            CMISS.FieldParameterSetTypes.VALUES,
                            1, deriv_idx + 1, node.id + offset, comp_idx + 1)
            mesh3D.save('{0}/supine.mesh'.format(op.results_dir))
    
    print 'Program successfully completed.'

if __name__ == "__main__":
    run()
