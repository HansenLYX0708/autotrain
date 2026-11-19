"""Pin the current process to the efficiency cores of a hybrid Intel CPU (Windows).

Workaround for the host's i9-14900K: CPU-heavy pure-Python work (the offline
rotate-augment box search) corrupts the interpreter heap on its P-cores and
dies with 0xC0000005, while the same run completes on the E-cores. See
docs/LOCALIZATION_WORKFLOW.md. A no-op on non-Windows or non-hybrid CPUs.
"""
import ctypes
import os
import sys

_RECORD_SIZE = 32  # SYSTEM_CPU_SET_INFORMATION with the CpuSet union member


def efficiency_core_mask():
    """Affinity mask (processor group 0) of the lowest efficiency class, or None if not hybrid."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    needed = ctypes.c_ulong(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        return None
    buffer = ctypes.create_string_buffer(needed.value)
    if not kernel32.GetSystemCpuSetInformation(buffer, needed, ctypes.byref(needed), None, 0):
        return None
    raw, offset, cores = buffer.raw, 0, []
    while offset + _RECORD_SIZE <= needed.value:
        size = int.from_bytes(raw[offset:offset + 4], "little")
        if size < _RECORD_SIZE:
            break
        group = int.from_bytes(raw[offset + 12:offset + 14], "little")
        index, efficiency = raw[offset + 14], raw[offset + 18]
        if group == 0:
            cores.append((index, efficiency))
        offset += size
    classes = {efficiency for _, efficiency in cores}
    if len(classes) < 2:
        return None
    lowest = min(classes)
    return sum(1 << index for index, efficiency in cores if efficiency == lowest and index < 64)


def pin_to_efficiency_cores():
    """Apply efficiency_core_mask() to this process; returns the mask applied or None."""
    mask = efficiency_core_mask()
    if not mask:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        print("[cpu-affinity] SetProcessAffinityMask failed: {}".format(ctypes.get_last_error()), file=sys.stderr, flush=True)
        return None
    print("[cpu-affinity] pinned to efficiency cores, mask 0x{:X}".format(mask), flush=True)
    return mask
