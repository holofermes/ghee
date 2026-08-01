use std::env;

fn main() {
    println!("cargo:rerun-if-changed=ghee/_engine");
    println!("cargo:rerun-if-env-changed=GHEE_SYSROOT");
    let mut b = cc::Build::new();
    b.file("ghee/_engine/engine.c")
        .file("ghee/_engine/encoder.c")
        .include("ghee/_engine")
        .opt_level(3)
        .flag_if_supported("-funroll-loops");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "arm" {
        // Zero W is ARMv6+VFP2; arm toolchain baselines default to v7 and fault there.
        // zig spells the cpu with underscores, gcc/clang with dashes.
        b.flag_if_supported("-mcpu=arm1176jzf_s");
        b.flag_if_supported("-mcpu=arm1176jzf-s");
    }
    if let Ok(sr) = env::var("GHEE_SYSROOT") {
        let libdir = match arch.as_str() {
            "aarch64" => "aarch64-linux-gnu",
            "arm" => "arm-linux-gnueabihf",
            _ => "x86_64-linux-gnu",
        };
        b.flag(format!("-isystem{sr}/usr/include"));
        b.flag(format!("-isystem{sr}/usr/include/libdrm"));
        println!("cargo:rustc-link-search=native={sr}/usr/lib/{libdir}");
    } else {
        b.flag("-isystem/usr/include/libdrm");
    }
    b.compile("ghee_engine");
    for lib in ["drm", "gbm", "EGL", "GLESv2"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
