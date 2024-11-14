import multiprocessing
from subprocess import Popen, PIPE, STDOUT # For running system commands
import os
import glob
import re
import signal
import time

import argparse

def timeout(time):
    """Time =  number of seconds your want for timeout"""
    signal.signal(signal.SIGALRM, input)
    signal.alarm(time)

def display_options():
    print 'List models currently running (0)'
    print 'Kill all models (-1)'

def process_monitor(folder, pool, pool_outputs):
    """Process monitor"""
    print 'Monitor started'
    display_options()
    exit = False
    while (not exit):
        timeout(2) #Timesout the user input prompt below
        try:
            value = input('')
        except:
            pass
        else:
            if value==None:
                pass
            elif value == 0:
                fnames = running_processes(folder, list_processes=True)
                n_running_processors = len(fnames)
                display_options()
            elif value > 0 and value <=n_running_processors:
                try:
                    pid_file = open(fnames[value-1], 'r') 
                except:
                    print "Model no longer running"
                else:
                    print pid_file.read()
                    pid_file.close()
                display_options()
            elif value == -1:
                print "User triggered kill"
                pool.terminate()
                pool.join()
                running_processes(folder, extension='*.pid', list_processes=False, kill_process=True)
                exit = True
            else :
                print "invalid user input"
                display_options()
        fnames = running_processes(folder)
        n_running_processors = len(fnames)
        if n_running_processors == 0:
            time.sleep(2)
            fnames = running_processes(folder)
            n_running_processors = len(fnames)
            if n_running_processors == 0:
                print "All jobs finished"
                pool_outputs.get()
                pool.join()
                exit = True
      
def running_processes(folder, extension='*.running', list_processes=False, kill_process=False):
    fnames = glob.glob(folder+extension)
    for fname_idx, fname in enumerate(fnames):
        text = re.split('__|./|\.{0}'.format(extension),fname)
        volunteer = text[2]
        parameter_set = text[3]
        if list_processes:
            print '  {0}, parameter set: {1} ({2})'.format(
                    volunteer,parameter_set, fname_idx+1)
        if extension == '*.pid' and kill_process:
            pid_file = open(fname, 'r') 
            pid = pid_file.read()
            print '  {0}, parameter set: {1}, pid: {2}'.format(
                                volunteer,parameter_set, pid)
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except:
                pass
            os.remove(fname)
            text = re.split('\.{0}'.format(extension),fname)
            running_fname = text[0]+'.running'
            try:
                os.remove(running_fname)
            except:
                pass
    return fnames

def perform_task(args):
    # Note that input_arg corresponds to an item in the inputs list defined in the main script
    folder, volunteer, parameters_set = args
    print 'Running: {0}, parameter set: {1}'.format(volunteer, parameters_set)
    debug = False
    if debug:
        script = 'tail -f test.txt'
    else:
        script = 'python prone_to_supine.py -v {0} -p {1} -os'.format(volunteer, parameters_set)

    run_fname = '{0}{1}__{2}.running'.format(folder,volunteer,parameters_set)
    stdout_fname = '{0}{1}__{2}.stdout'.format(folder,volunteer,parameters_set)
    stderr_fname = '{0}{1}__{2}.stderr'.format(folder,volunteer,parameters_set)
    pid_fname = '{0}{1}__{2}.pid'.format(folder,volunteer,parameters_set)

    run_file = open(run_fname, 'w')
    run_file.close()

    command = Popen(['{0} 1>{1} 2>{2}'.format(
        script, stdout_fname, stderr_fname)], preexec_fn=os.setsid,shell=True)
    pid_file = open(pid_fname, 'w')
    pid_file.write('{0}'.format(command.pid))
    pid_file.close()
    command.wait() # Wait till process completes
    os.remove(pid_fname)
    os.remove(run_fname)
    time.sleep(1)
    return 'finished'

def start_process():
    pass
    #print 'Starting', multiprocessing.current_process().name

if __name__ == '__main__':
    pool_size = 1
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                maxtasksperchild=1)
    monitor_folder = './multiprocessing_results_set2/'
    if not os.path.exists(monitor_folder):
        os.makedirs(monitor_folder)

    debug = False
    if debug:
        folder_args = [monitor_folder, monitor_folder]
        volunteer_args = ['VL00048', 'VL00048']
        parameter_set_args = [4,4]
    else:
        folder_args = []
        volunteer_args = []
        parameter_set_args = []
        volunteer_list = []
        fnames = glob.glob('../../data/volunteer/ribcages/*.mesh')
        for fname in fnames:
            volunteer = re.split('_|/',fname)[5]
            volunteer_list.append(volunteer) 

        for parameter_set_value in range(5):
            for volunteer in volunteer_list:
                folder_args.append(monitor_folder)
                volunteer_args.append(volunteer)
                parameter_set_args.append(parameter_set_value)

    args = zip(folder_args,volunteer_args,parameter_set_args)
    pool_outputs = pool.map_async(perform_task, args)
    pool.close() # no more tasks

    process_monitor(monitor_folder, pool, pool_outputs)


