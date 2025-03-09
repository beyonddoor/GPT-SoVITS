
import os

is_hooked = False

def hook_popen():
    import subprocess
    original_popen = subprocess.Popen
    def _hooked_popen(*args, **kwargs):
        print(f"---------> subprocess.Popen called with args: {args}, kwargs: {kwargs}")
        return original_popen(*args, **kwargs)  # Call the original Popen
    subprocess.Popen = _hooked_popen

def hook_os_system():
    origin_open = os.open
    def _hook_os_system(*args, **kwargs):
        print("---> open", args, kwargs)
        return origin_open(*args, **kwargs)
    os.open = _hook_os_system

def hook_proc():
    global is_hooked
    if is_hooked:
        return
    is_hooked = True
    hook_popen()
    hook_os_system()

hook_proc()
