extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let llama_cpp_include_dir = env::var("LLAMA_CPP_INCLUDEDIR")
        .unwrap_or_else(|_| "../llama.cpp/".into());
    let llama_cpp_lib_dir = env::var("LLAMA_CPP_LIBDIR")
        .unwrap_or_else(|_| "../llama.cpp/".into());

    println!("cargo:rustc-link-lib=static=ggml_static");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-search={}", llama_cpp_lib_dir);
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG_PATH");

    #[allow(deprecated)]
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))

        .whitelist_type("llama_model_params")
        .whitelist_function("llama_.*")

        //.blacklist_function("llama_dump_timing_info_yaml")
        //.blacklist_type("FILE")
        //.blacklist_type("_IO_.*")

        .opaque_type("FILE")

        .anon_fields_prefix("anon")
        .clang_arg(format!("-I{}", llama_cpp_include_dir))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
