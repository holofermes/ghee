//! A scene param becomes a parameter_t view, and the C engine binds that view.
//! Rust owns the storage behind it for as long as the engine reads it.

use core::ffi::c_int;
use std::ffi::CString;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::ffi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum CParameterType {
    Float = ffi::PARAM_TYPE_FLOAT,
    Int = ffi::PARAM_TYPE_INT,
    Vec2 = ffi::PARAM_TYPE_VEC2,
    Vec3 = ffi::PARAM_TYPE_VEC3,
    Vec4 = ffi::PARAM_TYPE_VEC4,
}

/// The C view points at this per-type struct.
///
/// The field holds a raw pointer, and never the Box itself. A move retags a Box.
/// The retag invalidates every raw pointer to it, including the copy in
/// `Parameter.value`. Drop reclaims the allocation.
enum Owned {
    Float(*mut ffi::FloatParam),
    Int(*mut ffi::IntParam),
    Vec(*mut ffi::VecParam),
}

pub struct CParameter {
    param: ffi::Parameter,
    owned: Owned,
    /// The bytes behind `param.name`. A move of the CString leaves its heap
    /// buffer, and the pointer, where they are.
    name: CString,
}

impl Drop for CParameter {
    fn drop(&mut self) {
        // The C engine keeps no pointer to this storage after a call returns.
        unsafe {
            match self.owned {
                Owned::Float(p) => drop(Box::from_raw(p)),
                Owned::Int(p) => drop(Box::from_raw(p)),
                Owned::Vec(p) => drop(Box::from_raw(p)),
            }
        }
    }
}

impl CParameter {
    pub fn as_ptr(&self) -> *const ffi::Parameter {
        &self.param
    }

    /// Name the uniform for this param.
    pub fn set_c_name(&mut self, name: &str) -> Result<()> {
        self.name =
            CString::new(name).map_err(|_| anyhow!("parameter name {name:?} holds a NUL"))?;
        self.param.name = self.name.as_ptr();
        Ok(())
    }
}

fn as_float(value: &Value) -> Result<f64> {
    match value {
        Value::Number(n) => match n.as_f64() {
            Some(f) => Ok(f),
            None => bail!("parameter value is not a real number"),
        },
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => bail!("parameter value is not a real number"),
    }
}

fn as_int(value: &Value) -> Result<c_int> {
    match value {
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => Ok(i as c_int),
            (None, Some(f)) => Ok(f.trunc() as c_int),
            _ => bail!("parameter value is not an integer"),
        },
        Value::Bool(b) => Ok(c_int::from(*b)),
        _ => bail!("parameter value is not an integer"),
    }
}

/// `n` numbers, each a float.
fn as_floats(value: &Value, n: usize) -> Result<[f32; 4]> {
    let mut out = [0.0f32; 4];
    match value {
        Value::Array(a) if a.len() == n => {
            for (slot, v) in out.iter_mut().zip(a) {
                *slot = as_float(v)? as f32;
            }
            Ok(out)
        }
        _ => bail!("parameter value is not {n} numbers"),
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

pub fn create_c_parameter(param_type: &str, value: &Value) -> Result<CParameter> {
    let (ty, owned, cvalue) = match param_type {
        "float" => {
            let p = Box::into_raw(Box::new(ffi::FloatParam {
                value: as_float(value)? as f32,
            }));
            (
                CParameterType::Float,
                Owned::Float(p),
                ffi::ParamValue { float_param: p },
            )
        }
        "int" => {
            let p = Box::into_raw(Box::new(ffi::IntParam {
                value: as_int(value)?,
            }));
            (
                CParameterType::Int,
                Owned::Int(p),
                ffi::ParamValue { int_param: p },
            )
        }
        // The engine has no bool type. glUniform1i sets a bool uniform.
        "bool" => {
            let p = Box::into_raw(Box::new(ffi::IntParam {
                value: c_int::from(truthy(value)),
            }));
            (
                CParameterType::Int,
                Owned::Int(p),
                ffi::ParamValue { int_param: p },
            )
        }
        // A color is a vec3 uniform.
        "vec2" | "vec3" | "vec4" | "color" => {
            let (ty, n) = match param_type {
                "vec2" => (CParameterType::Vec2, 2),
                "vec4" => (CParameterType::Vec4, 4),
                _ => (CParameterType::Vec3, 3),
            };
            let p = Box::into_raw(Box::new(ffi::VecParam {
                value: as_floats(value, n)?,
            }));
            (ty, Owned::Vec(p), ffi::ParamValue { vec_param: p })
        }
        _ => bail!("unknown parameter type: {param_type}"),
    };
    let name = CString::default();
    Ok(CParameter {
        param: ffi::Parameter {
            name: name.as_ptr(),
            ty: ty as c_int,
            value: cvalue,
        },
        owned,
        name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_vector_param_carries_its_components() {
        for (ty, want, value) in [
            ("vec2", CParameterType::Vec2, json!([0.5, -1.0])),
            ("vec3", CParameterType::Vec3, json!([0.5, -1.0, 2.0])),
            ("color", CParameterType::Vec3, json!([0.5, -1.0, 2.0])),
            ("vec4", CParameterType::Vec4, json!([0.5, -1.0, 2.0, 0.25])),
        ] {
            let cp = create_c_parameter(ty, &value).unwrap();
            assert_eq!(cp.param.ty, want as c_int, "{ty}");
            let got = unsafe { (*cp.param.value.vec_param).value };
            let n = value.as_array().unwrap().len();
            for (g, v) in got.iter().zip(value.as_array().unwrap()) {
                assert_eq!(f64::from(*g), v.as_f64().unwrap(), "{ty}");
            }
            assert!(got[n..].iter().all(|g| *g == 0.0), "{ty}");
        }
        let e = create_c_parameter("vec3", &json!([0.0, 0.0]))
            .err()
            .map(|e| e.to_string());
        assert_eq!(e.as_deref(), Some("parameter value is not 3 numbers"));
    }

    #[test]
    fn bool_reaches_the_c_abi_as_an_int() {
        let cp = create_c_parameter("bool", &json!(true)).unwrap();
        assert_eq!(cp.param.ty, CParameterType::Int as c_int);
        assert_eq!(unsafe { (*cp.param.value.int_param).value }, 1);
        let cp = create_c_parameter("bool", &json!(false)).unwrap();
        assert_eq!(unsafe { (*cp.param.value.int_param).value }, 0);
    }

    #[test]
    fn float_and_int_carry_their_value() {
        let cp = create_c_parameter("float", &json!(0.25)).unwrap();
        assert_eq!(cp.param.ty, CParameterType::Float as c_int);
        assert_eq!(unsafe { (*cp.param.value.float_param).value }, 0.25);
        // An int param truncates a float value toward zero.
        let cp = create_c_parameter("int", &json!(3.7)).unwrap();
        assert_eq!(unsafe { (*cp.param.value.int_param).value }, 3);
    }

    #[test]
    fn a_named_parameter_survives_a_move() {
        let mut cp = create_c_parameter("float", &json!(2.0)).unwrap();
        cp.set_c_name("gain").unwrap();
        let moved = cp;
        assert_eq!(unsafe { (*moved.param.value.float_param).value }, 2.0);
        let name = unsafe { std::ffi::CStr::from_ptr(moved.param.name) };
        assert_eq!(name.to_str().unwrap(), "gain");
    }
}
