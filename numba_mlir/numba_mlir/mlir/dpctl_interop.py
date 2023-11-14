# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils import readenv

from collections import namedtuple

DEFAULT_DEVICE = readenv("NUMBA_MLIR_DEFAULT_DEVICE", str, "")

DeviceCaps = namedtuple(
    "DeviceCaps",
    [
        "filter_string",
        "spirv_major_version",
        "spirv_minor_version",
        "has_fp16",
        "has_fp64",
    ],
)

try:
    import dpctl
    from dpctl.tensor import usm_ndarray

    _is_dpctl_available = True
except ImportError:
    _is_dpctl_available = False

if _is_dpctl_available:
    import numba
    import numpy as np
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils, config, types, typing

    from numba.core.pythonapi import box, unbox, NativeValue

    from numba.extending import register_model
    from numba.np import numpy_support

    from numba.core.datamodel.models import StructModel

    from .target import typeof_impl
    from . import array_type

    try:
        from numba_dpex.core.types.usm_ndarray_type import USMNdArray as OtherUSMNdArray

        # TODO: add to numba-dpex USMNdArray
        def array_get_device_caps(obj):
            if hasattr(obj, "_caps"):
                return obj._caps

            device = obj.device
            if not device:
                return None

            caps = _get_device_caps(dpctl.SyclDevice(device))
            setattr(obj, "_caps", caps)

            return caps

        setattr(OtherUSMNdArray, "get_device_caps", array_get_device_caps)
    except ImportError:
        OtherUSMNdArray = None

    def _get_device_caps(device):
        return DeviceCaps(
            filter_string=device.filter_string,
            spirv_major_version=1,
            spirv_minor_version=2,
            has_fp16=device.has_aspect_fp16,
            has_fp64=device.has_aspect_fp64,
        )

    class USMNdArrayBaseType(array_type.FixedArray):
        """
        Type class for DPPY arrays.
        """

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            fixed_dims,
            readonly=False,
            name=None,
            aligned=True,
            filter_string=None,
            device=None,
        ):
            super(USMNdArrayBaseType, self).__init__(
                dtype,
                ndim,
                layout,
                fixed_dims,
                readonly=readonly,
                name=name,
                aligned=aligned,
            )

            self.filter_string = filter_string
            self.device = device
            self.caps = None

        @property
        def key(self):
            # Do not add a device, as it alreadfy covered by filter_string
            return super().key + (self.filter_string,)

        @property
        def box_type(self):
            return np.ndarray

        def is_precise(self):
            return self.dtype.is_precise()

        def get_device_caps(self):
            if self.caps:
                return self.caps

            device = self.device
            if not device:
                return None

            caps = _get_device_caps(device)
            self.device = None
            self.caps = caps

            return caps

    class USMNdArrayModel(StructModel):
        def __init__(self, dmm, fe_type):
            ndim = fe_type.ndim
            members = [
                ("meminfo", types.MemInfoPointer(fe_type.dtype)),
                ("parent", types.pyobject),
                ("nitems", types.intp),
                ("itemsize", types.intp),
                ("data", types.CPointer(fe_type.dtype)),
                ("shape", types.UniTuple(types.intp, ndim)),
                ("strides", types.UniTuple(types.intp, ndim)),
            ]
            super(USMNdArrayModel, self).__init__(dmm, fe_type, members)

    class USMNdArrayType(USMNdArrayBaseType):
        """
        USMNdArrayType(dtype, ndim, layout, usm_type,
                        readonly=False, name=None,
                        aligned=True)
        creates Numba type to represent ``dpctl.tensor.usm_ndarray``.
        """

        def __init__(
            self,
            dtype,
            ndim,
            layout,
            usm_type,
            fixed_dims,
            readonly=False,
            name=None,
            aligned=True,
            filter_string=None,
            device=None,
        ):
            self.usm_type = usm_type
            # This name defines how this type will be shown in Numba's type dumps.
            name = "USM:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
            super(USMNdArrayType, self).__init__(
                dtype,
                ndim,
                layout,
                fixed_dims,
                readonly=readonly,
                name=name,
                filter_string=filter_string,
                device=device,
            )

        def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
            if dtype is None:
                dtype = self.dtype
            if ndim is None:
                ndim = self.ndim
            if layout is None:
                layout = self.layout
            if readonly is None:
                readonly = not self.mutable
            return USMNdArrayType(
                dtype=dtype,
                ndim=ndim,
                layout=layout,
                usm_type=self.usm_type,
                fixed_dims=(None,) * ndim,
                readonly=readonly,
                aligned=self.aligned,
                filter_string=self.filter_string,
                device=self.device,
            )

    register_model(USMNdArrayType)(USMNdArrayModel)

    @typeof_impl.register(usm_ndarray)
    def typeof_usm_ndarray(val, c):
        """
        This function creates the Numba type (USMNdArrayType) when a usm_ndarray is passed.
        """
        try:
            dtype = numpy_support.from_dtype(val.dtype)
        except NotImplementedError:
            raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
        layout = numpy_support.map_layout(val)
        readonly = False

        device = val.device.sycl_device
        filter_string = device.filter_string
        fixed_dims = array_type.get_fixed_dims(val.shape)
        return USMNdArrayType(
            dtype,
            val.ndim,
            layout,
            val.usm_type,
            fixed_dims,
            readonly=readonly,
            filter_string=filter_string,
            device=device,
        )

    def adapt_sycl_array_from_python(pyapi, ary, ptr):
        assert pyapi.context.enable_nrt
        fnty = ir.FunctionType(ir.IntType(32), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, name="nmrtUnboxSyclInterface")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
        return pyapi.builder.call(fn, (ary, ptr))

    @unbox(USMNdArrayType)
    def unbox_array(typ, obj, c):
        nativearycls = c.context.make_array(typ)
        nativeary = nativearycls(c.context, c.builder)
        aryptr = nativeary._getpointer()

        ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
        errcode = adapt_sycl_array_from_python(c.pyapi, obj, ptr)
        failed = cgutils.is_not_null(c.builder, errcode)

        # Handle error
        with c.builder.if_then(failed, likely=False):
            c.pyapi.err_set_string(
                "PyExc_TypeError",
                "can't unbox array from PyObject into "
                "native value.  The object maybe of a "
                "different type",
            )
        return NativeValue(c.builder.load(aryptr), is_error=failed)

    def _get_filter_string(array):
        if isinstance(array, usm_ndarray):
            return array.device.sycl_device.filter_string

        return None

    def check_usm_ndarray_args(args):
        devs = set(s for s in map(_get_filter_string, args) if s is not None)
        if len(devs) > 1:
            dev_names = ", ".join(devs)
            err_str = f"usm_ndarray arguments have incompatibe devices: {dev_names}"
            raise ValueError(err_str)

    def get_default_device():
        if DEFAULT_DEVICE:
            device = dpctl.SyclDevice(DEFAULT_DEVICE)
        else:
            device = dpctl.select_default_device()

        return _get_device_caps(device)

else:  # _is_dpctl_available
    USMNdArrayType = None  # dummy

    def check_usm_ndarray_args(args):
        # dpctl is not loaded, nothing to do
        pass

    def get_default_device():
        # TODO: deprecated
        if DEFAULT_DEVICE:
            device_name = DEFAULT_DEVICE
        else:
            device_name = "level_zero:gpu:0"

        return DeviceCaps(
            filter_string=device_name,
            spirv_major_version=1,
            spirv_minor_version=2,
            has_fp16=True,
            has_fp64=False,
        )
