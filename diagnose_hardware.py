"""Print CPU, RAM, and PyTorch accelerator availability."""

import os

import psutil
import torch


def gibibytes(byte_count: int) -> float:
    """Convert bytes to GiB."""
    return byte_count / 1024**3


def main() -> None:
    """Print resources relevant to selecting SAM patch and batch sizes."""
    memory = psutil.virtual_memory()
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {gibibytes(memory.total):.2f} GiB")
    print(f"Available RAM: {gibibytes(memory.available):.2f} GiB")

    if not torch.cuda.is_available():
        print("CUDA: unavailable")
        return

    device_id = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_id)
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    print(f"CUDA device: {torch.cuda.get_device_name(device_id)}")
    print(f"Total VRAM: {gibibytes(properties.total_memory):.2f} GiB")
    print(f"Allocated VRAM: {gibibytes(allocated):.2f} GiB")
    print(f"Reserved VRAM: {gibibytes(reserved):.2f} GiB")
    print(f"Process: {os.getpid()}")


if __name__ == "__main__":
    main()

