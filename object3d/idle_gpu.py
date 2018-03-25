import os
import subprocess
import json

__all__ = ["GPUStatNotFoundError", "NoIdleGPUsError", "try_limit"]

class GPUStatNotFoundError(Exception):
    pass


class NoIdleGPUsError(Exception):
    pass


def try_limit(memory=None):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    idle_gpu = _get_idle_gpu()
    if idle_gpu is None:
        if memory is None:
            raise NoIdleGPUsError("No true idle GPU")
        idle_gpu = _get_idle_gpu(memory)
        if idle_gpu is None:
            raise NoIdleGPUsError("No idle GPU with enough memory({})".format(memory))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle_gpu)


def _get_idle_gpu(memory=None):
    try:
        gpustat_json = subprocess.check_output(["gpustat", "--json"])
    except FileNotFoundError as e:
        raise GPUStatNotFoundError(e)

    gpustat = json.loads(gpustat_json.decode("utf-8"))

    if memory:
        for gpu in gpustat["gpus"]:
            index = int(gpu["index"])
            total = int(gpustat["memory.total"])
            used = int(gpustat["memory.used"])
            if total - used > memory:
                return index
    else:
        for gpu in gpustat["gpus"]:
            index = int(gpu["index"])
            if len(gpu["processes"]) == 0:
                return index

    return None
