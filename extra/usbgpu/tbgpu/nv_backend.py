from __future__ import annotations
from tinygrad.runtime.ops_nv import NVDevice, NVProgram

# This file is the only place in the standalone shim that still reaches into tinygrad's NV bring-up path.
# It can be replaced later with a fully local tbgpu/NV implementation without changing the public shim API.

def open_device(ordinal:int) -> NVDevice:
  return NVDevice(f"NV:{ordinal}")

def load_program(device:NVDevice, name:str, image:bytes) -> NVProgram:
  return NVProgram(device, name, image)
