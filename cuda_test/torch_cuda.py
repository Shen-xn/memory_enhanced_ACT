import sys

import torch


def main():
    print("=== PyTorch CUDA Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("\nStatus: CUDA unavailable")
        if "+cpu" in torch.__version__ or torch.version.cuda is None:
            print("Reason: current PyTorch is CPU-only, not a CUDA build.")
            print("Hint: install a CUDA-enabled torch that matches your driver and CUDA runtime.")
        else:
            print("Reason: PyTorch has CUDA support, but no usable GPU was exposed to this process.")
            print("Hint: check NVIDIA driver, CUDA runtime, and whether the GPU is visible in this environment.")
        return

    current_idx = torch.cuda.current_device()
    print(f"Current device index: {current_idx}")
    print(f"Current device name: {torch.cuda.get_device_name(current_idx)}")

    x = torch.tensor([1.0, 2.0], device="cuda")
    print(f"Tensor device: {x.device}")
    print("Status: CUDA ready")


if __name__ == "__main__":
    main()
