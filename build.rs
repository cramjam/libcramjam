fn main() {
    // See note in Cargo.toml
    if cfg!(target_pointer_width = "64") {
        #[cfg(feature = "isal-if-supported")]
        println!("cargo:rustc-cfg=feature=\"isal-is-supported\"");

        #[cfg(feature = "use-system-isal")]
        println!("cargo:rustc-cfg=feature=\"isal-rs/use-system-isal\"");

        #[cfg(feature = "isal-static")]
        println!("cargo:rustc-cfg=feature=\"isal-rs/static\"");

        #[cfg(feature = "isal-shared")]
        println!("cargo:rustc-cfg=feature=\"isal-rs/shared\"");
    } else {
        let msg = "feature set but ISA-L not supported on 32 bit systems.";

        #[cfg(feature = "isal-if-supported")]
        println!("cargo:warning='isal-if-supported' {}", msg);

        #[cfg(feature = "use-system-isal")]
        println!("cargo:warning='use-system-isal' {}", msg);

        #[cfg(feature = "isal-static")]
        println!("cargo:warning='isal-static' {}", msg);

        #[cfg(feature = "isal-shared")]
        println!("cargo:warning='isal-shared' {}", msg);
    }
}
