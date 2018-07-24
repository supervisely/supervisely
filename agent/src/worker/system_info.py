# coding: utf-8

import os.path as osp
import platform
import subprocess

import psutil
import docker


# Some functions below perform dirty parsing of corresponding utils output.
# @TODO: They shall be replaced with more portable implementations.

# echo q | htop -C | aha --line-fix | html2text -width 999 | grep -v "F1Help" | grep -v "xml version=" > file.txt
# echo q | nvidia-smi | aha --line-fix | html2text -width 999 | grep -v "F1Help" | grep -v "xml version=" > fnvsmi.txt


def _proc_run(in_args, strip=True, timeout=2):
    compl = subprocess.run(in_args, stdout=subprocess.PIPE, check=True, timeout=timeout)
    rows = compl.stdout.decode('utf-8').split('\n')
    if strip:
        rows = [x.strip() for x in rows]
    return rows


def _catch_silently(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _val_unit(s):
    s = s.strip()
    for i, c in enumerate(s):
        if not c.isdigit():
            return int(s[:i]), s[i:]
    return None


def parse_cpuinfo():
    rows = _proc_run(['cat', '/proc/cpuinfo'])
    logical_count = sum((1 for r in rows if r.startswith('processor')))
    physical_count = len(set((r.split()[-1] for r in rows if r.startswith('physical id'))))
    models = list(set((r.split(':')[-1] for r in rows if r.startswith('model name'))))
    res = {
        'models': models,
        'count': {
            'logical_cores': logical_count,
            'physical_cpus': physical_count,
        },
    }
    return res


def parse_meminfo():
    rows = _proc_run(['cat', '/proc/meminfo'])
    spl_colon = [x.split(':') for x in rows]
    dct = {z[0]: z[1] for z in spl_colon if len(z) > 1}

    def to_bytes(name):
        val, unit = dct[name].split()
        val = int(val)
        if unit == 'B':
            return val
        elif unit == 'kB':
            return val * 2**10
        elif unit == 'mB':
            return val * 2 ** 20
        elif unit == 'gB':
            return val * 2 ** 30
        raise RuntimeError('Unknown unit.')

    res = {
        'memory_B': {
            'physical': to_bytes('MemTotal'),
            'swap': to_bytes('SwapTotal'),
        },
    }
    return res


# def parse_nvsmi():
#     # /usr/local/cuda/bin/nvcc --version cannot be applied: docker cuda
#     devs = {}
#
#     name_rows = _proc_run(['nvidia-smi', '-L'])
#     for nr in name_rows:
#         spl_colon = nr.split(':')
#         if len(spl_colon) < 2:
#             continue
#         gpu_id = int(spl_colon[0].split()[-1])
#         model = spl_colon[1].split('(')[0].strip()
#         devs[gpu_id] = {'model': model}
#
#     rows = _proc_run(['nvidia-smi'])
#
#     def get_row_blocks():
#         out_lst = []
#         for r in rows:
#             r_part = r[1:4]
#             if (r_part == '===') or (r_part == '---'):
#                 if len(out_lst) > 0:
#                     yield out_lst
#                     out_lst = []
#             else:
#                 out_lst.append(r)
#
#     def try_parse_gpu_info(raw_strs):
#         if len(raw_strs) != 2:
#             return
#         by_cols = [z.strip('|').split('|') for z in raw_strs]
#         if any((len(z) != 3 for z in by_cols)):
#             return
#         if any((z in raw_strs[0] for z in ['Name', 'Persistence-M', 'Bus-Id', 'Volatile'])):
#             return
#
#         the_gpu_id = int(by_cols[0][0].split()[0])
#         temp = by_cols[1][0].split()[1]
#         mems = by_cols[1][1].split('/')
#         util = by_cols[1][2].split()[0]
#
#         def to_dct(val_unit_str):
#             v, u = _val_unit(val_unit_str)
#             return {'value': v, 'unit': u}
#
#         to_upd = {
#             'temperature': to_dct(temp),
#             'memory': {
#                 'used': to_dct(mems[0]),
#                 'total': to_dct(mems[1]),
#             },
#             'utilization': to_dct(util),
#         }
#         devs[the_gpu_id].update(to_upd)
#
#     row_blocks = list(get_row_blocks())
#     for x in row_blocks:
#         try_parse_gpu_info(x)
#
#     return devs


def print_nvsmi_devlist():
    name_rows = _proc_run(['nvidia-smi', '-L'])
    return name_rows


def print_nvsmi():
    res_rows = _proc_run(['nvidia-smi'])
    return res_rows


def cpu_freq_MHZ():
    res = psutil.cpu_freq(percpu=False).max
    return res


def get_hw_info():
    res = {
        'psutil': {
            'cpu': {
                'count': {
                    'logical_cores': psutil.cpu_count(logical=True),
                    'physical_cores': psutil.cpu_count(logical=False),
                },
                # 'freq_max_MHZ': _catch_silently(cpu_freq_MHZ)
            },
            'memory_B': {
                'physical': psutil.virtual_memory()[0],
                'swap': psutil.swap_memory()[0],
            },
        },
        'platform': {
            'uname': platform.uname(),
        },
        'cpuinfo': _catch_silently(parse_cpuinfo),
        'meminfo': _catch_silently(parse_meminfo),
        'nvidia-smi': _catch_silently(print_nvsmi_devlist),
        # 'lshw_raw': _catch_silently(_proc_run, ['lshw'], strip=False),
    }
    return res


def get_load_info():
    vmem = psutil.virtual_memory()
    res = {
        'nvidia-smi': _catch_silently(print_nvsmi),
        'cpu_percent': psutil.cpu_percent(interval=0.1, percpu=True),  # @TODO: does the blocking call hit performance?
        'memory_B': {
            'total': vmem[0],
            'available': vmem[1],
        },
    }
    return res


def parse_du_hs(dir_path, timeout):
    du_res = _proc_run(['du', '-sb', dir_path], timeout=timeout)
    byte_str = du_res[0].split()[0].strip()
    byte_sz = int(byte_str)
    return byte_sz


def get_directory_size_bytes(dir_path, timeout=10):
    if not dir_path or not osp.isdir(dir_path):
        return 0
    res = _catch_silently(parse_du_hs, dir_path, timeout)
    if res is None:
        return -1  # ok, to indicate error
    return res


def _get_self_container_idx():
    rows = _proc_run(['cat', '/proc/self/cgroup'])
    s = rows[0]
    slash_pos = s.rfind('/')
    container_idx = s[slash_pos+1:]
    return container_idx


def _get_self_docker_image_digest():
    container_idx = _get_self_container_idx()
    dc = docker.from_env()
    self_cont = dc.containers.get(container_idx)
    self_img = self_cont.image
    self_img_digests = list(self_img.attrs['RepoDigests'])
    common_digests = set(x.split('@')[1] for x in self_img_digests)  # "registry.blah-blah.com@sha256:value"
    if len(common_digests) > 1:
        raise RuntimeError('Unable to determine unique image digest.')
    elif len(common_digests) == 0:
        return None
    else:
        res = common_digests.pop()
        return res


def get_self_docker_image_digest():
    return _catch_silently(_get_self_docker_image_digest)
