import glob
import re
import os
import numpy

results_folder = 'hpc2_results_set2/'
folder_names = glob.glob(results_folder + '/*')
volunteer_list = []
for folder_name in folder_names:
    volunteer = re.split(results_folder,folder_name)[1]
    volunteer_list.append(volunteer) 

f_v = []
c_v = []
u_v = []
parameter_set = 4
for v_string in sorted(volunteer_list):
    v = int(re.split('VL000',v_string)[1])
    if v in [20, 22, 26, 32, 38, 42, 43, 44, 53, 56, 60, 64, 67, 73, 76, 78,
              77, 70,55]:
        f_v.append(v)
    else:
        converged_fname = os.sep.join((results_folder,v_string,"{0}_{1}.finished".format(v_string, parameter_set)))
        if os.path.exists(converged_fname):
            converged_file = open(converged_fname, 'r') 
            err_code = converged_file.read() 
            if err_code == 'True':
                c_v.append(v)
            elif err_code == 'False':
                u_v.append(v)
print u_v
print c_v
print len(u_v)
print len(c_v)
success_rato = float(len(c_v))/float(len(c_v)+len(u_v))*100.
print success_rato


