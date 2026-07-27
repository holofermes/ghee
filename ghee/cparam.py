import ctypes
from enum import IntEnum

NAME_MAX = 255  # parameter_t.name is char[256]; the C side needs the terminator


class CParameterType(IntEnum):
    FLOAT = 0
    INT = 1
    VEC2 = 2
    VEC3 = 3
    VEC4 = 4


class CFloatParam(ctypes.Structure):
    _fields_ = [("value", ctypes.c_float)]


class CIntParam(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int)]


class CVec2Param(ctypes.Structure):
    _fields_ = [
        (
            "values",
            ctypes.c_float * 2,  # type: ignore
        )
    ]


class CVec3Param(ctypes.Structure):
    _fields_ = [
        (
            "values",
            ctypes.c_float * 3,  # type: ignore
        )
    ]


class CVec4Param(ctypes.Structure):
    _fields_ = [
        (
            "values",
            ctypes.c_float * 4,  # type: ignore
        )
    ]


class CParameterValue(ctypes.Union):
    _fields_ = [
        ("float_param", ctypes.POINTER(CFloatParam)),
        ("int_param", ctypes.POINTER(CIntParam)),
        ("vec2_param", ctypes.POINTER(CVec2Param)),
        ("vec3_param", ctypes.POINTER(CVec3Param)),
        ("vec4_param", ctypes.POINTER(CVec4Param)),
    ]


class CParameter(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("type", ctypes.c_int),
        ("value", CParameterValue),
    ]

    type: CParameterType


def set_c_name(param, name):
    """Name the uniform this parameter sets. Longer than the C field's payload and
    glGetUniformLocation would read past the terminator."""
    raw = name.encode("utf-8")
    if len(raw) > NAME_MAX:
        raise ValueError(
            f"parameter name is {len(raw)} bytes, the engine takes {NAME_MAX}: {name}"
        )
    param.name = raw


def create_c_parameter(param_data) -> CParameter:
    param = CParameter()
    param_type = param_data["type"]

    match param_type:
        case "float":
            param.type = CParameterType.FLOAT
            float_param = CFloatParam(value=float(param_data["value"]))
            param.value.float_param = ctypes.pointer(float_param)

        case "int":
            param.type = CParameterType.INT
            int_param = CIntParam(value=int(param_data["value"]))
            param.value.int_param = ctypes.pointer(int_param)

        case "bool":
            param.type = CParameterType.INT
            int_param = CIntParam(value=1 if param_data["value"] else 0)
            param.value.int_param = ctypes.pointer(int_param)

        case _:
            raise ValueError(f"unknown parameter type: {param_type}")

    return param
