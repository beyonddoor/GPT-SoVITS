
import os

from icecream import ic
log_debug = ic

is_hooked = False

def hook_popen():
    import subprocess
    origin_func = subprocess.Popen
    def _hooked_popen(*args, **kwargs):
        log_debug(f"==> subprocess.Popen", args, kwargs, os.environ)
        return origin_func(*args, **kwargs)
    subprocess.Popen = _hooked_popen

def hook_os_system():
    origin_func = os.system
    def _hook_os_system(*args, **kwargs):
        log_debug("==> os.system", args, kwargs, os.environ)
        return origin_func(*args, **kwargs)
    os.system = _hook_os_system

def hook_env():
    origin_func = os.environ.update
    def _hook_env(*args, **kwargs):
        log_debug("==> os.environ.update", args, kwargs)
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
