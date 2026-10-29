import shutil
import torch


def diagnostics(output_root):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Enable a Kaggle GPU accelerator.")
    free, total = torch.cuda.mem_get_info()
    disk = shutil.disk_usage(output_root)
    return {"gpu": torch.cuda.get_device_name(0), "vram_free_gb": free / 2**30,
            "vram_total_gb": total / 2**30, "disk_free_gb": disk.free / 2**30}
