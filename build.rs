use std::env;

fn main() {
    println!("cargo:rerun-if-changed=ghee/_engine");
    println!("cargo:rerun-if-changed=ghee/_freetype");
    println!("cargo:rerun-if-env-changed=GHEE_SYSROOT");
    let mut b = cc::Build::new();
    b.file("ghee/_engine/engine.c")
        .file("ghee/_engine/encoder.c")
        .include("ghee/_engine")
        .opt_level(3)
        .flag_if_supported("-funroll-loops");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "arm" {
        // The Pi Zero W CPU is ARMv6 with VFP2. An arm toolchain defaults to
        // ARMv7 and faults on that CPU.
        // zig writes the cpu name with underscores. gcc and clang write it with
        // dashes.
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
    freetype(&arch);
    for lib in ["drm", "gbm", "EGL", "GLESv2"] {
        println!("cargo:rustc-link-lib={lib}");
    }
}

/// FreeType 2.14.3 in vendor/freetype, the version that Pillow's wheels carry,
/// with the modules and options of ghee/_freetype, and ghee's calls into it.
fn freetype(arch: &str) {
    let mut b = cc::Build::new();
    let src = "vendor/freetype/src";
    for f in [
        "base/ftsystem.c",
        "base/ftinit.c",
        "base/ftdebug.c",
        "base/ftbase.c",
        "base/ftbitmap.c",
        "base/ftmm.c",
        "autofit/autofit.c",
        "truetype/truetype.c",
        "sfnt/sfnt.c",
        "smooth/smooth.c",
        "raster/raster.c",
        "psnames/psnames.c",
        "cff/cff.c",
        "psaux/psaux.c",
        "pshinter/pshinter.c",
    ] {
        b.file(format!("{src}/{f}"));
    }
    b.file("ghee/_freetype/glyphs.c")
        .include("ghee/_freetype")
        .include("vendor/freetype/include")
        .define("FT2_BUILD_LIBRARY", None)
        .define("FT_CONFIG_OPTIONS_H", "<ghee_ftoption.h>")
        .define("FT_CONFIG_MODULES_H", "<ghee_ftmodule.h>")
        .opt_level(2)
        .warnings(false);
    if arch == "arm" {
        b.flag_if_supported("-mcpu=arm1176jzf_s");
        b.flag_if_supported("-mcpu=arm1176jzf-s");
    }
    b.compile("ghee_freetype");
}
