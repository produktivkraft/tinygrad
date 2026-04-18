# TinyGPU

TinyGPU app lets you use AMD and NVIDIA GPUs on macOS over USB4/Thunderbolt with tinygrad.

## Requirements

- macOS (12.1+)
- USB4/Thunderbolt port
- A supported GPU (AMD RDNA3+ or NVIDIA Ampere+)

## Setup

### 1. Connect your GPU

Plug the supported GPU into your Mac over USB4/Thunderbolt.

### 2. Initiate the driver install

> **Note:** If tinygrad is cloned but not installed, run commands with `PYTHONPATH=.`

```bash
curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_tinygpu_osx.sh | sh
```

This downloads TinyGPU.app and triggers a system prompt to install the driver extension.

### 3. Enable the driver

You should see a system prompt: **"TinyGPU" would like to use a new driver extension**. Click **Open System Settings** and toggle TinyGPU on.

If you missed the prompt, go to **System Settings > General > Login Items & Extensions > Driver Extensions** and toggle TinyGPU on.

### 4. Compiler Setup

#### AMD

```bash
curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_hipcomgr_osx.sh | sh
```

#### NV

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you don't have it.

```bash
curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_nvcc_osx.sh | sh
```

Make sure `~/.local/bin` is on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 5. Use it!

```bash
DEV={AMD|NV} python3 -m tinygrad.llm
```

### CUDA shim on macOS

There is a standalone CUDA Driver API shim prototype under `extra/usbgpu/tbgpu` that talks to TinyGPU's NVIDIA path directly.
Use the vector-add demo to exercise the path:

```bash
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input cuda --launch-mode extra
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input ptx --launch-mode kernel_params
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input ptx --emit-cubin /tmp/vector_add.cubin
python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input cubin --cubin /tmp/vector_add.cubin
```

**Note:** Use `JITBEAM=2` to search for faster kernels (one-time search cost, results cached).
