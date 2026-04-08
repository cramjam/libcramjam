fn main() {
    use std::io::{Cursor, Read};
    fn read_dir_files(dir: std::path::PathBuf) -> Vec<u8> {
        let mut all_bytes = vec![];
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                all_bytes.extend(std::fs::read(entry.path()).unwrap());
            } else if entry.file_type().unwrap().is_dir() {
                all_bytes.extend(read_dir_files(entry.path()));
            }
        }
        all_bytes
    }
    use std::str::FromStr;
    let src = read_dir_files(std::path::PathBuf::from_str("./src").unwrap());
    println!("src dir size: {}", src.len());
    for level in [1, 3, 6, 9] {
        let mut out = Vec::new();
        libcramjam::zstd::compress(&mut Cursor::new(src.as_slice()), &mut out, Some(level), Some(src.len())).unwrap();
        let mut enc = zstd::stream::read::Encoder::new(src.as_slice(), level).unwrap();
        let mut c_out = Vec::new();
        enc.read_to_end(&mut c_out).unwrap();
        println!("L{}: ours={} ({:.2}%), c_zstd={} ({:.2}%)",
            level, out.len(), out.len() as f64 / src.len() as f64 * 100.0,
            c_out.len(), c_out.len() as f64 / src.len() as f64 * 100.0);
    }
}
