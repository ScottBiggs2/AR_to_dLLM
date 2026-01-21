import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

def main():
    print("Initializing Accelerator...")
    try:
        accelerator = Accelerator()
        print(f"Success! Process index: {accelerator.process_index}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Device: {accelerator.device}")
    except Exception as e:
        print(f"FAILED to initialize Accelerator: {e}")
        raise e

if __name__ == "__main__":
    main()
