extern crate nalgebra_glm as glm;
extern crate palette;
extern crate rand;
use palette::rgb::Rgb;
use rand::prelude::{Rng, ThreadRng};

const PI: f32 = 3.14159265;
const TAU: f32 = 2. * PI;

fn random_in_unit_disk(rng: &mut ThreadRng) -> glm::Vec3 {
    let theta: f32 = rng.gen_range(0., TAU);
    let r = rng.gen_range(0., 1.);

    glm::vec3(r * theta.cos(), r * theta.sin(), 0.)
}

fn ortho(v: &glm::Vec3) -> glm::Vec3 {
    if v.x.abs() > v.z.abs() {
        glm::vec3(-v.y, v.x, 0.)
    } else {
        glm::vec3(0., -v.z, v.y)
    }
}

fn get_biased_sample(dir: &glm::Vec3, power: f32, rng: &mut ThreadRng) -> glm::Vec3 {
    let d = glm::normalize(&dir);
    let o1 = glm::normalize(&ortho(&d));
    let o2 = glm::normalize(&glm::cross(&d, &o1));
    let th = rng.gen_range(0., TAU);
    let mut y: f32 = rng.gen_range(0., 1.);
    y = y.powf(1. / (power + 1.));
    let oneminus = (1. - y * y).sqrt();

    th.cos() * oneminus * o1 + th.sin() * oneminus * o2 + y * d
}

fn cosine_weighted_sample(dir: &glm::Vec3, rng: &mut ThreadRng) -> glm::Vec3 {
    get_biased_sample(dir, 1., rng)
}

fn random_on_unit_hemi(dir: &glm::Vec3, rng: &mut ThreadRng) -> glm::Vec3 {
    get_biased_sample(dir, 0., rng)
}

struct Ray {
    o: glm::Vec3,
    dir: glm::Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> glm::Vec3 {
        self.o + t * self.dir
    }
}

#[derive(Debug, Copy, Clone)]
struct DiffuseMat {
    albedo: Rgb,
}

#[derive(Debug, Copy, Clone)]
struct MetalMat {
    albedo: Rgb,
    roughness: f32,
}

#[derive(Debug, Copy, Clone)]
struct GlassMat {
    ior: f32,
}

#[derive(Debug, Copy, Clone)]
enum Material {
    Diffuse(DiffuseMat),
    Metallic(MetalMat),
    Glass(GlassMat),
}

struct Hit {
    p: glm::Vec3,
    n: glm::Vec3,
    m: Material,
    t: f32,
    front_face: bool,
}

impl Hit {
    pub fn new() -> Self {
        Hit {
            p: glm::vec3(0., 0., 0.),
            n: glm::vec3(0., 1., 0.),
            m: Material::Diffuse(DiffuseMat {
                albedo: Rgb::new(0., 0., 0.),
            }),
            t: TMAX,
            front_face: false,
        }
    }
}

struct Sphere {
    center: glm::Vec3,
    radius: f32,
    material: Material,
}

impl Sphere {
    pub fn new(center: glm::Vec3, radius: f32, material: Material) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }

    pub fn intersect(&self, r: &Ray, h: &mut Hit) -> bool {
        let oc: glm::Vec3 = r.o - self.center;
        let b = glm::dot(&oc, &r.dir);
        let c = oc.magnitude_squared() - self.radius.powi(2);
        let d = b * b - c;
        if d < 0. {
            return false;
        };

        let t1 = -b - d.sqrt();
        let t2 = -b + d.sqrt();
        let t = if t1 < TMIN { t2 } else { t1 };

        let p = r.at(t);
        let mut n = p - self.center;
        let front_face = glm::dot(&r.dir, &n) < 0.;
        n = if front_face { n } else { -n };
        n /= self.radius;

        if t < TMIN || t > TMAX {
            return false;
        };

        if t < h.t {
            *h = Hit {
                p,
                n,
                m: self.material,
                t,
                front_face,
            };
        };

        return true;
    }
}

pub struct Camera {
    o: glm::Vec3,
    lower_left: glm::Vec3,
    horizontal: glm::Vec3,
    vertical: glm::Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new((width, height): (f32, f32)) -> Self {
        let aperature = 1.;
        let fov: f32 = 60.;
        let theta = fov.to_radians();
        let h = (theta / 2.).tan() * 2.;
        let aspect = width / height;
        let w = h * aspect;

        let o = glm::vec3(0., 1.3, -5.);
        let lookat = glm::vec3(0., 0., 0.);
        let focal_depth = glm::length(&(o - lookat));
        let up = glm::vec3(0., 1., 0.);
        let to = glm::normalize(&(o - lookat));
        let u = glm::normalize(&glm::cross(&up, &to));
        let v = glm::cross(&to, &u);

        let horizontal = w * u * focal_depth;
        let vertical = h * v * focal_depth;
        let lower_left = o - horizontal / 2. - vertical / 2. - focal_depth * to;
        let lens_radius = aperature / 2.;

        Camera {
            o,
            lower_left,
            horizontal,
            vertical,
            lens_radius,
        }
    }

    fn get_ray(&self, uv: glm::Vec2, rng: &mut ThreadRng) -> Ray {
        let rd: glm::Vec3 = random_in_unit_disk(rng) * self.lens_radius;
        let offset = glm::vec3(uv.x, uv.y, 0.).component_mul(&rd);
        Ray {
            o: self.o + offset,
            dir: glm::normalize(
                &(self.lower_left + uv.x * self.horizontal + uv.y * self.vertical
                    - self.o
                    - offset),
            ),
        }
    }
}

const TMIN: f32 = 0.001;
const TMAX: f32 = 100000000.;
const MAX_ITER: u8 = 5;

fn raycast(r: &Ray, h: &mut Hit) -> bool {
    let mut is_hit = false;

    let pl: Sphere = Sphere {
        center: glm::vec3(0., -201., 0.),
        radius: 200.,
        material: Material::Diffuse(DiffuseMat {
            albedo: Rgb::new(0.8, 0.8, 0.8),
        }),
    };

    let s = Sphere {
        center: glm::vec3(0., 0., 0.),
        radius: 1.,
        material: Material::Diffuse(DiffuseMat {
            albedo: Rgb::new(0.9, 0.3, 0.9),
        }),
    };
    let s2 = Sphere::new(
        glm::vec3(2., 0., 0.),
        1.,
        Material::Metallic(MetalMat {
            albedo: Rgb::new(0.8, 0.8, 0.2),
            roughness: 0.3,
        }),
    );

    is_hit = pl.intersect(r, h) || is_hit;
    is_hit = s.intersect(r, h) || is_hit;
    is_hit = s2.intersect(r, h) || is_hit;

    return is_hit;
}

fn shade_diffuse(r: &mut Ray, h: &Hit, mat: DiffuseMat, rng: &mut ThreadRng) -> glm::Vec3 {
    r.dir = cosine_weighted_sample(&r.dir, rng);
    r.o = h.p;

    glm::vec3(mat.albedo.red, mat.albedo.green, mat.albedo.blue)
}

fn shade_metallic(r: &mut Ray, h: &Hit, mat: MetalMat, rng: &mut ThreadRng) -> glm::Vec3 {
    let refl = glm::reflect_vec(&r.dir, &h.n);
    let jitter = random_on_unit_hemi(&h.n, rng) * mat.roughness;

    r.o = h.p;
    r.dir = glm::normalize(&(refl + jitter));

    glm::vec3(mat.albedo.red, mat.albedo.green, mat.albedo.blue)
}

fn schlick(cos: f32, ior: f32) -> f32 {
    let mut r0 = (1. - ior) / (1. + ior);
    r0 *= r0;

    r0 + (1. - r0) * (1. - cos).powi(5)
}

fn shade_glass(r: &mut Ray, h: &Hit, mat: GlassMat, rng: &mut ThreadRng) -> glm::Vec3 {
    r.o = h.p;
    let mut eta = mat.ior;
    if !h.front_face {
        eta = 1. / eta;
    };

    let cos = glm::dot(&-r.dir, &h.n);
    let sin = (1. - cos * cos).sqrt();

    let refl_prob = schlick(cos, mat.ior);
    let is_refl = rng.gen_range(0., 1.) < refl_prob;
    if is_refl || sin * eta > 1. {
        r.dir = glm::reflect_vec(&r.dir, &h.n);
        return glm::vec3(1., 1., 1.);
    };

    r.dir = glm::refract_vec(&r.dir, &h.n, eta);
    return glm::vec3(1., 1., 1.);
}

fn ray_color(r: &mut Ray, rng: &mut ThreadRng) -> glm::Vec3 {
    let mut h: Hit = Hit::new();
    let mut c = glm::vec3(1., 1., 1.);

    for _ in 0..MAX_ITER {
        h.t = TMAX;
        let is_hit = raycast(&r, &mut h);
        if is_hit {
            match h.m {
                Material::Diffuse(mat) => {
                    c.component_mul_assign(&shade_diffuse(r, &h, mat, rng));
                }
                Material::Metallic(mat) => c.component_mul_assign(&shade_metallic(r, &h, mat, rng)),
                Material::Glass(mat) => c.component_mul_assign(&shade_glass(r, &h, mat, rng)),
            }
        } else {
            let t = 0.5 * (r.dir.y + 1.);
            c = c.component_mul(&glm::lerp(
                &glm::vec3(1., 1., 1.),
                &glm::vec3(0.3, 0.7, 0.9),
                t,
            ));
            return c;
        }
    }

    return c;
}

pub fn raytrace(uv: glm::Vec2, cam: &Camera, rng: &mut ThreadRng) -> glm::Vec3 {
    let mut r = cam.get_ray(uv, rng);

    ray_color(&mut r, rng)
}
