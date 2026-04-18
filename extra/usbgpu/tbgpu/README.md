# tbgpu CUDA Shim

This directory contains a standalone CUDA Driver API shim prototype for macOS TinyGPU / tbgpu NVIDIA devices.
It is kept outside the main tinygrad runtime so the implementation, demo, and tests can evolve independently.

## Layout

- `cuda_shim.py`: Python CUDA Driver API shim that maps CUDA-style contexts, allocations, modules, and kernel launches onto TinyGPU's NV path.
- `nv_backend.py`: Thin dependency boundary around the low-level NV bring-up pieces reused from tinygrad today.
- `vector_add_demo.py`: End-to-end vector add demo for the shim. It keeps one CUDA C kernel and one PTX kernel source, and can optionally emit a cubin for inspection.

## Run the demo

```bash
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input cuda --launch-mode extra
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input ptx --launch-mode kernel_params
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input ptx --emit-cubin /tmp/vector_add.cubin
```

## Notes

- This code is isolated from `tinygrad/runtime/ops_cuda.py` on purpose.
- The shim is isolated from the main tinygrad CUDA runtime, but `nv_backend.py` still reuses low-level NVIDIA bring-up pieces from tinygrad's NV stack for now, because those are the parts that already know how to speak to TinyGPU's remote PCI path.
- `--kernel-input cuda` runs `CUDA C -> PTX`, then the shim compiles PTX to the execution image internally.
- `--kernel-input ptx` runs `PTX`, then the shim compiles PTX to the execution image internally.
- The `cuda` and `ptx` modes need `nvcc`, which on macOS usually means running `extra/setup_nvcc_osx.sh`.
