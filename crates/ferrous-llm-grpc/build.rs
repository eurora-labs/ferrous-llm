use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let proto_dir = PathBuf::from("./proto");
    let proto_files = std::fs::read_dir(&proto_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "proto"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();

    #[cfg(target_os = "windows")]
    {
        let common_dir = PathBuf::from("C:\\protoc\\include");
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            // .compile_well_known_types(true)
            .protoc_arg("--experimental_allow_proto3_optional")
            .compile_protos(&proto_files, &[proto_dir, common_dir])?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .protoc_arg("--experimental_allow_proto3_optional")
            .compile_protos(&proto_files, &[proto_dir])?;
    }
    Ok(())
}
