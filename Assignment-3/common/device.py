import shutil
import os
import torch


def diagnostics(output_root):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Enable a Kaggle GPU accelerator.")
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "vram_free_gb": free / 2**30, "vram_total_gb": total / 2**30})
    disk = shutil.disk_usage(output_root)
    ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
              if os.name == "posix" else None)
    return {"devices": devices, "host_ram_gb": ram_gb, "disk_free_gb": disk.free / 2**30}
