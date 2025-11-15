fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let has_python_feature = std::env::var_os("CARGO_FEATURE_PYTHON").is_some();

    if has_python_feature && target_os == "macos" {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
