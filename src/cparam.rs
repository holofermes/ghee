//! A scene param becomes the parameter_t view the C engine binds. Rust owns the
//! storage that view points at, for as long as the engine reads it.

use core::ffi::{c_char, c_int};

use anyhow::{bail, Result};
use serde_json::Value;

use crate::ffi;

/// parameter_t.name is char[256]. The C side needs the terminator byte.
pub const NAME_MAX: usize = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
#[allow(dead_code)] // The C enum declares the vector types. Nothing here produces one.
pub enum CParameterType {
    Float = ffi::PARAM_TYPE_FLOAT,
    Int = ffi::PARAM_TYPE_INT,
    Vec2 = ffi::PARAM_TYPE_VEC2,
    Vec3 = ffi::PARAM_TYPE_VEC3,
    Vec4 = ffi::PARAM_TYPE_VEC4,
}

/// The per-type struct the C view points at.
///
/// The field holds a raw pointer, and never the Box itself. A move retags a Box.
/// The retag invalidates every raw pointer to it, including the copy in
/// `Parameter.value`. Drop reclaims the allocation.
enum Owned {
    Float(*mut ffi::FloatParam),
    Int(*mut ffi::IntParam),
}

pub struct CParameter {
    param: ffi::Parameter,
    owned: Owned,
}

impl Drop for CParameter {
    fn drop(&mut self) {
        // The C engine keeps no pointer to this storage after a call returns.
        unsafe {
            match self.owned {
                Owned::Float(p) => drop(Box::from_raw(p)),
                Owned::Int(p) => drop(Box::from_raw(p)),
            }
        }
    }
}

impl CParameter {
    pub fn as_ptr(&self) -> *const ffi::Parameter {
        &self.param
    }

    /// Name the uniform this param sets. A longer name fills the C field, and
    /// glGetUniformLocation then reads past the terminator.
    pub fn set_c_name(&mut self, name: &str) -> Result<()> {
        let raw = name.as_bytes();
        if raw.len() > NAME_MAX {
            bail!(
                "parameter name is {} bytes, the engine takes {NAME_MAX}: {name}",
                raw.len()
            );
        }
        self.param.name = [0; 256];
        for (slot, b) in self.param.name.iter_mut().zip(raw) {
            *slot = *b as c_char;
        }
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
        _ => bail!("unknown parameter type: {param_type}"),
    };
    Ok(CParameter {
        param: ffi::Parameter {
            name: [0; 256],
            ty: ty as c_int,
            value: cvalue,
        },
        owned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn param_name_must_leave_room_for_the_terminator() {
        let mut cp = create_c_parameter("float", &json!(1.0)).unwrap();
        let name = "k".repeat(NAME_MAX);
        cp.set_c_name(&name).unwrap();
        let mut want = [0 as c_char; 256];
        for (slot, b) in want.iter_mut().zip(name.as_bytes()) {
            *slot = *b as c_char;
        }
        assert_eq!(cp.param.name, want);
        let long = "k".repeat(NAME_MAX + 1);
        assert_eq!(
            cp.set_c_name(&long).unwrap_err().to_string(),
            format!("parameter name is 256 bytes, the engine takes 255: {long}")
        );
    }

    #[test]
    fn vector_param_has_no_producer() {
        // A color and a vecN reach the shader as decomposed float uniforms.
        let e = create_c_parameter("vec3", &json!([0.0, 0.0, 0.0]))
            .err()
            .map(|e| e.to_string());
        assert_eq!(e.as_deref(), Some("unknown parameter type: vec3"));
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
        assert_eq!(moved.param.name[0], b'g' as c_char);
    }
}
