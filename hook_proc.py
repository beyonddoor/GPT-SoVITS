
import os

is_hooked = False

def hook_popen():
    import subprocess
    origin_func = subprocess.Popen
    def _hooked_popen(*args, **kwargs):
        print(f"---------> subprocess.Popen called with args: {args}, kwargs: {kwargs}")
        return origin_func(*args, **kwargs)  # Call the original Popen
    subprocess.Popen = _hooked_popen

def hook_os_system():
    origin_func = os.system
    def _hook_os_system(*args, **kwargs):
        print("---> system", args, kwargs)
        return origin_func(*args, **kwargs)
    os.system = _hook_os_system

def hook_env():
    origin_func = os.environ.update
    def _hook_env(*args, **kwargs):
        print("---> env update", args, kwargs)
        return origin_func(*args, **kwargs)
    os.environ.update = _hook_env

def hook_proc():
    global is_hooked
    if is_hooked:
        return
    is_hooked = True
    hook_popen()
    hook_os_system()
    hook_env()

hook_proc()
