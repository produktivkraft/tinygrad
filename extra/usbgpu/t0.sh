export LIBUSB_PATH="${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib/libusb-1.0.0.dylib"

# python extra/usbgpu/scan_pci.py
# python extra/usbgpu/debug.py

# https://docs.tinygrad.org/tinygpu/
# DEV=NV python3 tinygrad/apps/llm.py

# https://x.com/anemll/status/1985770450874745308
# proxy_all
# DEBUG=1 DEV=NV python3 test/test_tiny.py

DEBUG=3 DEV=NV python3 -c \
  "from tinygrad import Tensor;
N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N);
(a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2).realize()"

DEBUG=3 DEV=NV python3 extra/usbgpu/tbgpu/vector_add_demo.py --kernel-input cuda --launch-mode extra
