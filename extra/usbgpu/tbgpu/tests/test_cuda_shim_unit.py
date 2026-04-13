import pathlib, tempfile, sys, unittest
from unittest import mock

sys.path.insert(0, pathlib.Path(__file__).resolve().parents[4].as_posix())

from extra.usbgpu.tbgpu import cuda_shim
from extra.usbgpu.tbgpu import vector_add_demo as demo
from extra.usbgpu.tbgpu.vector_add_demo import VecAddArgs, _make_extra, _make_kernel_params

class TestCUDAShimHelpers(unittest.TestCase):
  def test_parse_vector_add_signature(self):
    sigs = cuda_shim._parse_ptx_signatures(demo.render_vector_add_ptx("sm_89"))
    self.assertIn("vector_add", sigs)
    self.assertEqual([(p.size, p.align) for p in sigs["vector_add"]], [(8, 8), (8, 8), (8, 8), (4, 4)])

  def test_render_vector_add_cuda(self):
    src = demo.render_vector_add_cuda()
    self.assertIn('extern "C" __global__ void vector_add', src)
    self.assertIn('c[idx] = a[idx] + b[idx];', src)

  def test_extract_extra_blob(self):
    args = VecAddArgs(0x11, 0x22, 0x33, 7)
    extra, _ = _make_extra(args)
    blob = cuda_shim._extract_extra_blob(extra)
    self.assertEqual(blob, demo._encode_args_blob(args))

  def test_marshal_kernel_params(self):
    args = VecAddArgs(0x11, 0x22, 0x33, 7)
    params, _ = _make_kernel_params(args)
    sig = cuda_shim._parse_ptx_signatures(demo.render_vector_add_ptx("sm_89"))["vector_add"]
    blob = cuda_shim._marshal_kernel_params(params, sig)
    self.assertEqual(blob, demo._encode_args_blob(args))

  def test_extra_matches_kernel_params_blob(self):
    args = VecAddArgs(0x11, 0x22, 0x33, 7)
    extra, _ = _make_extra(args)
    params, _ = _make_kernel_params(args)
    sig = cuda_shim._parse_ptx_signatures(demo.render_vector_add_ptx("sm_89"))["vector_add"]
    self.assertEqual(cuda_shim._extract_extra_blob(extra), cuda_shim._marshal_kernel_params(params, sig))

  def test_iter_chunk_args(self):
    launches = list(demo._iter_chunk_args(130, 64, 0x1000, 0x2000, 0x3000))
    self.assertEqual([(a.a, a.b, a.c, a.n) for a in launches], [
      (0x1000, 0x2000, 0x3000, 64),
      (0x1100, 0x2100, 0x3100, 64),
      (0x1200, 0x2200, 0x3200, 2),
    ])

  def test_load_kernel_image_cuda_path(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ptx_path, cubin_path = pathlib.Path(tmpdir) / "out.ptx", pathlib.Path(tmpdir) / "out.cubin"
      with mock.patch.object(demo, "compile_cuda_to_ptx", return_value=b"PTX"), \
           mock.patch.object(demo, "compile_ptx_to_cubin", return_value=b"CUBIN"):
        image = demo.load_kernel_image("sm_89", "cuda", emit_ptx=ptx_path.as_posix(), emit_cubin=cubin_path.as_posix())
      self.assertEqual(image, b"CUBIN")
      self.assertEqual(ptx_path.read_bytes(), b"PTX")
      self.assertEqual(cubin_path.read_bytes(), b"CUBIN")

  def test_load_kernel_image_cubin_path(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cubin_path = pathlib.Path(tmpdir) / "prebuilt.cubin"
      cubin_path.write_bytes(b"CUBIN")
      image = demo.load_kernel_image("sm_89", "cubin", cubin_path=cubin_path.as_posix())
      self.assertEqual(image, b"CUBIN")

if __name__ == '__main__':
  unittest.main()
