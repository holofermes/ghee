//! The version sites. tools/release.sh writes them in one pass.

use std::fs;

fn read(path: &str) -> String {
    let full = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&full).unwrap_or_else(|e| panic!("{full}: {e}"))
}

/// The value of the first `key = "value"` line of a TOML file.
fn field(path: &str, key: &str) -> String {
    let text = read(path);
    let head = format!("{key} = \"");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&head))
        .unwrap_or_else(|| panic!("{path}: no {key}"));
    line.split('"').nth(1).unwrap_or_default().to_string()
}

#[test]
fn the_two_crates_carry_one_version() {
    let version = field("Cargo.toml", "version");
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
    assert_eq!(version, field("python/Cargo.toml", "version"));
}

/// dpkg sorts `0.3.0-dev` after `0.3.0`, and `0.3.0~dev` before it. cargo-deb
/// takes no manifest key for the Debian version, so the workflow derives it and
/// every cargo-deb call carries it.
#[test]
fn every_deb_takes_the_debian_spelling_of_the_version() {
    let text = read(".github/workflows/debs.yml");
    let built = text.matches("cargo deb ").count();
    assert!(built > 0, "no deb is built");
    assert_eq!(built, text.matches("--deb-version").count());
    assert!(text.contains("tr '-' '~'"), "the ~ spelling is not derived");
}

/// A release builds from a released version alone, and the tag names it.
#[test]
fn the_release_workflows_check_the_tag() {
    for workflow in [".github/workflows/wheels.yml", ".github/workflows/debs.yml"] {
        let text = read(workflow);
        assert!(text.contains("github.event.release.tag_name"), "{workflow}");
        assert!(text.contains("no release builds from it"), "{workflow}");
    }
}
