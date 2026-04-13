from __future__ import annotations
import argparse, array, ctypes, pathlib, shutil, struct, subprocess, sys, tempfile

if __package__ in (None, ""):
  sys.path.insert(0, pathlib.Path(__file__).resolve().parents[3].as_posix())

from extra.usbgpu.tbgpu import cuda_shim as cuda
from tinygrad.runtime.support.c import del_an, init_c_var

KERNEL_NAME = "vector_add"

VECTOR_ADD_CUDA = r'''
extern "C" __global__ void vector_add(const float *a, const float *b, float *c, unsigned int n, unsigned int block_size) {
  // NOTE: On the standalone tbgpu/NV cubin path, reading %ntid.x has been observed to return 0.
  // Use host-provided block_size for idx math so multi-CTA launch remains correct.
  unsigned int idx = blockIdx.x * block_size + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
'''

VECTOR_ADD_PTX = r'''.version VERSION
.target TARGET
.address_size 64

.visible .entry vector_add(
  .param .u64 a,
  .param .u64 b,
  .param .u64 c,
  .param .u32 n,
  .param .u32 block_size
)
{
  .reg .pred %p<2>;
  .reg .f32 %f<4>;
  .reg .b32 %r<7>;
  .reg .b64 %rd<8>;

  ld.param.u64 %rd1, [a];
  ld.param.u64 %rd2, [b];
  ld.param.u64 %rd3, [c];
  ld.param.u32 %r1, [n];
  ld.param.u32 %r2, [block_size];

  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r3, %r2, %r4;
  setp.ge.u32 %p1, %r5, %r1;
  @%p1 bra DONE;

  mul.wide.u32 %rd4, %r5, 4;
  add.s64 %rd5, %rd1, %rd4;
  add.s64 %rd6, %rd2, %rd4;
  add.s64 %rd7, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  ld.global.f32 %f2, [%rd6];
  add.f32 %f3, %f1, %f2;
  st.global.f32 [%rd7], %f3;

DONE:
  ret;
}
'''

class VecAddArgs(ctypes.Structure):
  _fields_ = [("a", ctypes.c_uint64), ("b", ctypes.c_uint64), ("c", ctypes.c_uint64), ("n", ctypes.c_uint32), ("block_size", ctypes.c_uint32)]


def render_vector_add_cuda() -> str:
  return VECTOR_ADD_CUDA.strip() + "\n"


def _ptx_version_for_arch(arch:str) -> str:
  ver = int(arch.removeprefix("sm_"))
  if ver >= 120: return "8.7"
  if ver >= 89: return "7.8"
  return "7.5"


def render_vector_add_ptx(arch:str) -> bytes:
  return VECTOR_ADD_PTX.replace("TARGET", arch).replace("VERSION", _ptx_version_for_arch(arch)).encode()


def _require_nvcc() -> str:
  if (nvcc:=shutil.which("nvcc")) is None:
    raise RuntimeError("nvcc not found, run extra/setup_nvcc_osx.sh or use --kernel-input cubin with a prebuilt cubin")
  return nvcc


def _run_nvcc(arch:str, mode:str, src_path:pathlib.Path, out_path:pathlib.Path):
  _require_nvcc()
  proc = subprocess.run(["nvcc", f"-arch={arch}", mode, "-o", out_path.as_posix(), src_path.as_posix()],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
  if proc.returncode != 0:
    raise RuntimeError(f"nvcc failed for {mode} on {src_path.name}: {(proc.stderr or proc.stdout).decode('utf-8', 'ignore').strip()}")


def compile_cuda_to_ptx(cuda_src:str, arch:str) -> bytes:
  with tempfile.TemporaryDirectory(prefix="tbgpu_vecadd_cuda_") as tmpdir:
    src_path, out_path = pathlib.Path(tmpdir) / f"{KERNEL_NAME}.cu", pathlib.Path(tmpdir) / f"{KERNEL_NAME}.ptx"
    src_path.write_text(cuda_src)
    _run_nvcc(arch, "-ptx", src_path, out_path)
    return out_path.read_bytes()


def compile_ptx_to_cubin(ptx:bytes, arch:str) -> bytes:
  return cuda._compile_ptx_to_cubin(ptx, arch)


def _write_if_requested(path:str|None, data:bytes):
  if path is not None: pathlib.Path(path).write_bytes(data)


def load_kernel_image(arch:str, kernel_input:str, cubin_path:str|None=None, emit_ptx:str|None=None, emit_cubin:str|None=None) -> bytes:
  if kernel_input == "cuda":
    ptx = compile_cuda_to_ptx(render_vector_add_cuda(), arch)
    cubin = compile_ptx_to_cubin(ptx, arch)
    _write_if_requested(emit_ptx, ptx)
    _write_if_requested(emit_cubin, cubin)
    return cubin

  if kernel_input == "ptx":
    ptx = render_vector_add_ptx(arch)
    cubin = compile_ptx_to_cubin(ptx, arch)
    _write_if_requested(emit_ptx, ptx)
    _write_if_requested(emit_cubin, cubin)
    return cubin

  if kernel_input == "cubin":
    if cubin_path is None: raise ValueError("--cubin is required when --kernel-input cubin")
    cubin = pathlib.Path(cubin_path).read_bytes()
    _write_if_requested(emit_cubin, cubin)
    return cubin

  raise ValueError(f"unsupported kernel input {kernel_input}")


def _check(status:int):
  if status != 0:
    err = ctypes.POINTER(ctypes.c_char)()
    cuda.cuGetErrorString(status, ctypes.byref(err))
    raise RuntimeError(f"CUDA shim error {status}: {ctypes.string_at(err).decode()}")


def _buffer_ptr(buf:array.array) -> int:
  return ctypes.addressof((ctypes.c_float * len(buf)).from_buffer(buf))


def _encode_args_blob(args:VecAddArgs) -> bytes:
  return struct.pack("<QQQII", args.a, args.b, args.c, args.n, args.block_size)


def _make_extra(args:VecAddArgs):
  arg_blob = _encode_args_blob(args)
  arg_buf = ctypes.create_string_buffer(arg_blob)
  arg_size = ctypes.c_size_t(len(arg_blob))
  extra = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(arg_buf, ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p), ctypes.c_void_p(0))
  return extra, (arg_buf, arg_size)


def _make_kernel_params(args:VecAddArgs):
  scalars = [ctypes.c_uint64(args.a), ctypes.c_uint64(args.b), ctypes.c_uint64(args.c), ctypes.c_uint32(args.n), ctypes.c_uint32(args.block_size)]
  params = (ctypes.c_void_p * len(scalars))(*[ctypes.addressof(v) for v in scalars])
  return params, scalars


def run_vector_add(size:int=256, block_size:int=64, launch_mode:str="extra", kernel_input:str="ptx", cubin_path:str|None=None,
                   emit_ptx:str|None=None, emit_cubin:str|None=None) -> array.array:
  if block_size <= 0: raise ValueError(f"block_size must be positive, got {block_size}")
  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))
  _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))

  arch = f"sm_{major.value}{minor.value}"
  kernel_image = load_kernel_image(arch, kernel_input, cubin_path=cubin_path, emit_ptx=emit_ptx, emit_cubin=emit_cubin)
  module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
  func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))

  a = array.array('f', (float(i) for i in range(size)))
  b = array.array('f', (float(2 * i + 1) for i in range(size)))
  out = array.array('f', [0.0] * size)

  nbytes = size * ctypes.sizeof(ctypes.c_float)
  d_a = del_an(cuda.CUdeviceptr)()
  d_b = del_an(cuda.CUdeviceptr)()
  d_out = del_an(cuda.CUdeviceptr)()

  try:
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_a, _buffer_ptr(a), nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b, _buffer_ptr(b), nbytes, None))

    block = (block_size, 1, 1)
    if size > 0:
      grid = ((size + block_size - 1) // block_size, 1, 1)
      # Keep block_size in args for parity with the CUDA source and handwritten PTX kernel path.
      args = VecAddArgs(d_a.value, d_b.value, d_out.value, size, block_size)
      if launch_mode == "kernel_params":
        params, keepalive = _make_kernel_params(args)
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, params, None))
      else:
        extra, keepalive = _make_extra(args)
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, None, extra))

    _check(cuda.cuCtxSynchronize())
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out), d_out, nbytes))
  finally:
    for ptr in [d_out, d_b, d_a]:
      if ptr.value not in (None, 0): cuda.cuMemFree_v2(ptr)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)

  expected = array.array('f', (x + y for x, y in zip(a, b)))
  for got, exp in zip(out, expected):
    if abs(got - exp) > 1e-5: raise AssertionError(f"vector add mismatch: {got} != {exp}")
  return out


def main():
  parser = argparse.ArgumentParser(description="Run a vector-add CUDA kernel through the TinyGPU CUDA shim")
  parser.add_argument("--size", type=int, default=256)
  parser.add_argument("--block-size", type=int, default=64)
  parser.add_argument("--launch-mode", choices=["extra", "kernel_params"], default="extra")
  parser.add_argument("--kernel-input", choices=["cuda", "ptx", "cubin"], default="ptx")
  parser.add_argument("--cubin", help="Path to a prebuilt cubin when --kernel-input cubin is selected")
  parser.add_argument("--emit-ptx", help="Optional path to write the PTX used for this run")
  parser.add_argument("--emit-cubin", help="Optional path to write the cubin used for this run")
  args = parser.parse_args()
  run_vector_add(size=args.size, block_size=args.block_size, launch_mode=args.launch_mode, kernel_input=args.kernel_input, cubin_path=args.cubin,
                 emit_ptx=args.emit_ptx, emit_cubin=args.emit_cubin)
  print(f"vector add ok, size={args.size}, launch_mode={args.launch_mode}, kernel_input={args.kernel_input}")


if __name__ == "__main__":
  main()
