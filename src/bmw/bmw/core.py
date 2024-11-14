import os
import argparse
from bmw import utils

def setup(p):
    params = utils.Params(p)

    if not os.path.exists(params.results_dir):
        os.makedirs(params.results_dir)
    if params.offscreen:
        fig = None
        viewer = None
    else:
        from morphic import viewer
        #viewer.set_offscreen(True)
        viewer = viewer

    print ('Volunteer: {0}'.format(params.volunteer))
    print ('Offscreen: {0}'.format(params.offscreen))

    return params, viewer

class volunteer_setup():

    def __init__(self, mesh_dir, results_dir, volunteer, parameters, offscreen):
        self.mesh_dir = mesh_dir
        self.results_dir = results_dir
        self.volunteer = volunteer
        self.parameters = parameters

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if offscreen:
            fig = None
            viewer = None
        else:
            from morphic import viewer
            #viewer.set_offscreen(True)
        self.viewer = viewer

        print ('Volunteer: {0}'.format(volunteer))
        print ('Parameters: {0}'.format(parameters))
        print ('Offscreen: {0}'.format(offscreen))

    def add_fig(self, label):
        if self.viewer is not None:
            fig = self.viewer.Figure(label)
        else:
            fig = None
        return fig

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-v", "--volunteer", help="volunteer, e.g. VL00048",
                        type=str)
    parser.add_argument("-p", "--parameter_set", help="the parameter set to solve, e.g. 0",
                        type=int)
    parser.add_argument("-os", "--offscreen", action='store_true', default=False, help="Offscreen rendering, e.g. on or off")
    arguments = parser.parse_args()

    return arguments


