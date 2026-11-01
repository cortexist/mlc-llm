"""Attach frontend modules after cache dispatch has populated external_mods."""

import tvm_tirx


@tvm_tirx.transform.module_pass(opt_level=0, name="AttachExternalModules")
class AttachExternModules:
    def __init__(self, modules):
        self.modules = modules

    def transform_module(self, mod, _ctx):
        # TVM 629cc65820 calls append on the immutable ffi.Array stored in attrs.
        # Copy it first, preserving modules already supplied by the KV cache.
        attrs = dict(mod.attrs) if mod.attrs else {}
        modules = list(attrs.get("external_mods", []))
        modules.extend(module.load() for module in self.modules)
        return mod.with_attr("external_mods", modules)
