import ctypes
from enum import IntEnum


class CParameterType(IntEnum):
    FLOAT = 0
    INT = 1
    VEC2 = 2
    VEC3 = 3
    VEC4 = 4
    TEXTURE = 5


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


class CTextureParam(ctypes.Structure):
    _fields_ = [("value", ctypes.c_char * 256)]


class CParameterValue(ctypes.Union):
    _fields_ = [
        ("float_param", ctypes.POINTER(CFloatParam)),
        ("int_param", ctypes.POINTER(CIntParam)),
        ("vec2_param", ctypes.POINTER(CVec2Param)),
        ("vec3_param", ctypes.POINTER(CVec3Param)),
        ("vec4_param", ctypes.POINTER(CVec4Param)),
        ("texture_param", ctypes.POINTER(CTextureParam)),
    ]


class CParameter(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("type", ctypes.c_int),
        ("value", CParameterValue),
    ]

    type: CParameterType


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

        case "vec2":
            param.type = CParameterType.VEC2
            vec2_param = CVec2Param(values=(ctypes.c_float * 2)(*param_data["value"]))
            param.value.vec2_param = ctypes.pointer(vec2_param)

        case "vec3":
            param.type = CParameterType.VEC3
            vec3_param = CVec3Param(values=(ctypes.c_float * 3)(*param_data["value"]))
            param.value.vec3_param = ctypes.pointer(vec3_param)

        case "vec4":
            param.type = CParameterType.VEC4
            vec4_param = CVec4Param(values=(ctypes.c_float * 4)(*param_data["value"]))
            param.value.vec4_param = ctypes.pointer(vec4_param)

        case _:
            raise ValueError(f"unknown parameter type: {param_type}")

    return param
