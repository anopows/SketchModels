import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from datetime import datetime
from pytz import timezone
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ptensors
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.python.tools import inspect_checkpoint
import io
from contextlib import redirect_stdout

def get_time(location = "Europe/Zurich"):
    fmt = "%Y-%m-%d_%H:%M"
    return datetime.now(timezone(location)).strftime(fmt)

def limit_cpu(num_threads):
    return tf.ConfigProto(intra_op_parallelism_threads=num_threads,inter_op_parallelism_threads=num_threads)

def in_jupyter():
    try:
        cfg = get_ipython().config # Name error if not exists --> CPython,..
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell' # ipython or jupyter_
    except NameError:
        return False

def names_from_ckpt(model_folder, name=''):
    if not os.path.isdir(model_folder):
        raise Exception("Directory '{}' does not exist".format(model_folder))
    latest_ckpt = tf.train.latest_checkpoint(model_folder)
    if latest_ckpt is None:
        raise Exception("No checkpoints found")
    ptensors(latest_ckpt, all_tensors=False, tensor_name=name)

def list_devices():
    return list_local_devices()

def get_logger(file_name, log_dir='../log', print_to_stdout=True):
    logging.basicConfig(level=logging.INFO, filename="{}/{}.log".format(log_dir,file_name),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    if print_to_stdout:
        logging.getLogger().addHandler(logging.StreamHandler())
    return logging

def plot_histogram(names, values, num_choices=None, figsize=(16,10)):
    # Ugly but other functions not dependent on these imports. TODO: refactor into other module
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.set(rc={'figure.figsize':figsize})
    sns.set(color_codes=True)
    for name,val in zip(names, values):
        if num_choices:
            val = np.random.choice(val, num_choices)
        val = np.log10(abs(val)+1e-20)
        sns.distplot(val, label=name)
    plt.legend()

# from https://stackoverflow.com/a/39649614
def tf_print(tensor, transform=None, message="", precision=2, linewidth=150, suppress=False):
    np.set_printoptions(precision=precision, suppress=suppress, linewidth=linewidth)
    # Insert a custom python operation into the graph that does nothing but print a tensors value 
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(message, x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res

def shapes(var, indent=0):
    def first_name(string):
        def normalchar(c):
            if c.isalnum():
                return c
            else:
                return ' '
        return ''.join(map(normalchar, string)).split()[0]
        
    name = first_name(str(var))
    start_str = "_"*indent
    
    if hasattr(var, 'shape'):
        print(start_str + "shape: " + str(var.shape), end=' ')
        if hasattr(var, 'dtype'): print(var.dtype)
        else: print()
    elif hasattr(var, '__iter__'):
        print(start_str + '|' + name)
        for v in var: shapes(v, indent+1)
    elif hasattr(var, '__len__'):
        print(start_str + "length: " + str(len(var)))
    else:
        print(start_str, "Cant print: ", name)


def get_ckpt_vars(path, path_is_folder=True, name='', with_data=False):
    if path_is_folder: # pick the latest checkpoint
        path = tf.train.latest_checkpoint(path)
    with io.StringIO() as buf, redirect_stdout(buf):
        inspect_checkpoint.print_tensors_in_checkpoint_file(path, name, with_data)
        entries = buf.getvalue().split('\n' )
        entries = filter(None, entries)
        entries = [e.split(' ') for e in entries]
        return entries

# TODO: fix bugs
def compare_checkpt_vars(path, path_is_folder=True, name=''):
    ckpt_names  = [v[0]   for v in get_ckpt_vars(path, path_is_folder, name=name)]
    scope_names = [v[:-2] for v in tf.trainable_variables(scope=name)]

    print("===========")
    print("Names only in checkpoint:")
    for ckpt_name in ckpt_names: 
        if ckpt_name not in scope_names:
            print(ckpt_name)
    print("-----------")
    print("Names only in current scope:")
    for scope_name in scope_names:
        if scope_name not in ckpt_names:
            print(scope_name)
    print("===========")
    return ckpt_names, scope_names