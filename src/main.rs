mod trace;
use trace::{DiffuseMat, GlassMat, HittableList, Material, MetalMat, Sphere, BVH};

extern crate image;
extern crate nalgebra_glm as glm;
extern crate palette;
use palette::{rgb::Rgb, Pixel, Srgb};

use rand::prelude::{Rng, ThreadRng};

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

fn create_scene(rng: &mut ThreadRng) -> BVH {
    let s = create_hittable_list(rng);

    BVH::from_hittable_list(&s, rng)
}

fn create_hittable_list(rng: &mut ThreadRng) -> HittableList {
    let mut s = HittableList::new();

    s.add(Box::new(Sphere::new(
        glm::vec3(0., -200., 0.),
        200.,
        Material::Diffuse(trace::DiffuseMat {
            albedo: Rgb::new(0.8, 0.8, 0.8),
        }),
    )));

    s.add(Box::new(Sphere::new(
        glm::vec3(0., 1.5, 0.),
        3.,
        Material::Diffuse(DiffuseMat {
            albedo: Rgb::new(0.9, 0.3, 0.9),
        }),
    )));

    s.add(Box::new(Sphere::new(
        glm::vec3(-8., 1.5, 0.),
        3.,
        Material::Metallic(MetalMat {
            albedo: Rgb::new(0.8, 0.8, 0.2),
            roughness: 0.3,
        }),
    )));

    for i in -5..5 {
        for j in -5..5 {
            let r = rng.gen_range(0., 1.);
            let g = rng.gen_range(0., 1.);
            let b = rng.gen_range(0., 1.);

            s.add(Box::new(Sphere::new(
                glm::vec3(i as f32, 0.125, j as f32),
                0.25,
                Material::Diffuse(DiffuseMat {
                    albedo: Rgb::new(r, g, b),
                }),
            )));
        }
    }

    s
}

fn main() {
    SimpleLogger::new().init().unwrap();

    let width = 800;
    let height = 600;

    let mut imgbuf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::new(width, height);
    let mut drawbuf = vec![vec![glm::vec3(0., 0., 0.); width as usize]; height as usize];

    let mut passes = 0;
    let max_pass = 200;
    let pb = ProgressBar::new(max_pass);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({eta})")
            .progress_chars("#>-"),
    );

    let cam = trace::Camera::new((width as f32, height as f32));
    let mut rng = rand::thread_rng();

    let num_threads = num_cpus::get() - 2;
    // let mut threads: Vec<std::thread::JoinHandle<_>> = Vec::with_capacity(num_threads);

    eprintln!("Ray-tracing using {} threads", num_threads);

    let default_scene = create_scene(&mut rng);

    for _ in 0..max_pass {
        for (x, y, _) in imgbuf.enumerate_pixels_mut() {
            let uv = glm::vec2(x as f32 / width as f32, 1. - y as f32 / height as f32);

            drawbuf[y as usize][x as usize] += trace::raytrace(uv, &cam, &default_scene, &mut rng);
        }
        passes += 1;
        pb.inc(1);

        if passes == 2 {
            save(passes, &mut imgbuf, &drawbuf);
        };

        if passes % 100 == 0 {
            save(passes, &mut imgbuf, &drawbuf);
        }
    }
    pb.finish();

    save(max_pass as i32, &mut imgbuf, &drawbuf);

    debug!("done");
}
