mod trace;

extern crate image;
extern crate nalgebra_glm as glm;
extern crate palette;
use palette::{Pixel, Srgb};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use simple_logger::SimpleLogger;

use indicatif::{ProgressBar, ProgressStyle};

fn save(
    pass: i32,
    imgbuf: &mut image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    drawbuf: &Vec<Vec<glm::Vec3>>,
) {
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let p = drawbuf[y as usize][x as usize] / (pass as f32);
        let c = Srgb::from_components((p.x, p.y, p.z));
        let raw = Srgb::into_raw(c.into_format());
        *pixel = image::Rgb(raw);
    }

    imgbuf
        .save_with_format(format!("out_{}.png", pass), image::ImageFormat::Png)
        .unwrap();
}

fn main() {
    SimpleLogger::new().init().unwrap();

    let width = 400;
    let height = 300;

    let mut imgbuf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::new(width, height);
    let mut drawbuf = vec![vec![glm::vec3(0., 0., 0.); width as usize]; height as usize];

    let mut passes = 0;
    let max_pass = 10;
    let pb = ProgressBar::new(max_pass);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({eta})")
            .progress_chars("#>-"),
    );

    let cam = trace::Camera::new((width as f32, height as f32));
    let mut rng = rand::thread_rng();

    for _ in 0..max_pass {
        for (x, y, _) in imgbuf.enumerate_pixels_mut() {
            let uv = glm::vec2(x as f32 / width as f32, 1. - y as f32 / height as f32);

            drawbuf[y as usize][x as usize] += trace::raytrace(uv, &cam, &mut rng);
        }
        passes += 1;
        pb.inc(1);

        if passes % 100 == 0 {
            save(passes, &mut imgbuf, &drawbuf);
        }
    }
    pb.finish();

    save(max_pass as i32, &mut imgbuf, &drawbuf);

    debug!("done");
}
