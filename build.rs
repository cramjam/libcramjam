fn main() {
    // See note in Cargo.toml
    if cfg!(target_pointer_width = "32") {
        let msg = "feature set but ISA-L not supported on 32 bit systems.";

        #[cfg(feature = "use-system-isal")]
        println!("cargo:warning='use-system-isal' {}", msg);

        #[cfg(feature = "isal-static")]
        println!("cargo:warning='isal-static' {}", msg);

        #[cfg(feature = "isal-shared")]
        println!("cargo:warning='isal-shared' {}", msg);
    }
}
