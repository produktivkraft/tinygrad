import ctypes, ctypes.util, os, pathlib, shutil, subprocess, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[4]
REAL_E2E = os.getenv("CUDA_SHIM_E2E") == "1"
NVCC_READY = shutil.which("nvcc") is not None


def _has_gpuocelot() -> bool:
  paths = [ctypes.util.find_library("gpuocelot")] if ctypes.util.find_library("gpuocelot") is not None else []
  paths += ["libgpuocelot.so", "/usr/local/lib/libgpuocelot.so", "libgpuocelot.dylib", "/usr/local/lib/libgpuocelot.dylib",
            "/opt/homebrew/lib/libgpuocelot.dylib"]
  for path in paths:
    try: ctypes.CDLL(path)
    except OSError: continue
    else: return True
  return False

MOCKGPU_READY = _has_gpuocelot()

@unittest.skipUnless((REAL_E2E or MOCKGPU_READY) and NVCC_READY, "real TinyGPU hardware or MOCKGPU+libgpuocelot plus nvcc is required")
class TestTBGPUCUDAShimE2E(unittest.TestCase):
  def _base_env(self):
    env = os.environ.copy()
    if not REAL_E2E: env["MOCKGPU"] = "1"
    return env

  def _run_demo(self, *extra_args:str):
    proc = subprocess.run([sys.executable, "extra/usbgpu/tbgpu/vector_add_demo.py", "--size", "128", *extra_args],
                          cwd=ROOT, env=self._base_env(), capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

  def test_vector_add_cuda(self):
    self._run_demo("--kernel-input", "cuda", "--launch-mode", "extra")

  def test_vector_add_ptx(self):
    self._run_demo("--kernel-input", "ptx", "--launch-mode", "kernel_params")

  def test_vector_add_cubin(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cubin_path = pathlib.Path(tmpdir) / "vector_add.cubin"
      self._run_demo("--kernel-input", "ptx", "--emit-cubin", cubin_path.as_posix())
      self._run_demo("--kernel-input", "cubin", "--cubin", cubin_path.as_posix())

if __name__ == '__main__':
  unittest.main()
