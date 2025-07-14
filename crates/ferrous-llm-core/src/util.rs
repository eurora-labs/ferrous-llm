#[cfg(feature = "dynamic-image")]
pub mod dynamic_image {
    use base64::{Engine, engine::general_purpose::STANDARD as B64};
    use image::{DynamicImage, ImageFormat};
    use std::io::Cursor;

    pub fn image_to_base64(img: &DynamicImage) -> String {
        // 1. Encode image to bytes once.
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("image encoding failed");

        // 2. Pre-size the output string to avoid reallocation.
        let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
        B64.encode_string(&bytes, &mut out);
        out
    }
}
