import os
import ctypes
import traceback


def get_calling_script():
    """Get the calling script from the traceback stack"""
    stack = traceback.extract_stack()

    script_path = None
    for trace in stack:
        if trace[2] == '<module>':
            script_path = trace[0]
    if script_path is None:
        for trace in stack:
            if '/run_' in trace[0]:
                script_path = trace[0]
    if script_path is None:
        script_path = stack[-1][0]  # default

    script_name = os.path.basename(script_path)
    if script_name[:14] == '<ipython-input':
        script_name = '<ipython>'
    if script_name[-3:] == '.py':
        script_name = script_name[:-3]

    return script_name


def positive_hash(obj):
    return ctypes.c_size_t(hash(obj)).value
