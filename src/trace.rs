extern crate itertools;
extern crate nalgebra_glm as glm;
extern crate palette;
extern crate rand;
use itertools::izip;
use palette::rgb::Rgb;
use rand::prelude::{Rng, ThreadRng};
use std::cmp::Ordering;
use std::f32::consts::*;

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

pub struct Ray {
    o: glm::Vec3,
    dir: glm::Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> glm::Vec3 {
        self.o + t * self.dir
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DiffuseMat {
    pub albedo: Rgb,
}

#[derive(Debug, Copy, Clone)]
pub struct MetalMat {
    pub albedo: Rgb,
    pub roughness: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct GlassMat {
    pub ior: f32,
}

#[derive(Debug, Copy, Clone)]
pub enum Material {
    Diffuse(DiffuseMat),
    Metallic(MetalMat),
    Glass(GlassMat),
}

pub struct Hit {
    pub p: glm::Vec3,
    pub n: glm::Vec3,
    pub m: Material,
    pub t: f32,
    pub front_face: bool,
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

pub trait Hittable {
    fn intersect(&self, r: &Ray, h: &mut Hit) -> bool;
    fn bounding_box(&self) -> Option<AABB>;
    fn box_clone(&self) -> Box<dyn Hittable>;
}

impl Clone for Box<dyn Hittable> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Sphere {
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
}

impl Hittable for Sphere {
    fn intersect(&self, r: &Ray, h: &mut Hit) -> bool {
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

    fn bounding_box(&self) -> Option<AABB> {
        Some(AABB {
            min: self.center - glm::vec3(self.radius, self.radius, self.radius),
            max: self.center + glm::vec3(self.radius, self.radius, self.radius),
        })
    }

    fn box_clone(&self) -> Box<dyn Hittable> {
        Box::new(*self)
    }
}

#[derive(Clone)]
pub struct AABB {
    pub min: glm::Vec3,
    pub max: glm::Vec3,
}

impl AABB {
    pub fn new() -> Self {
        AABB {
            min: glm::vec3(0., 0., 0.),
            max: glm::vec3(0., 0., 0.),
        }
    }

    pub fn create(min: &glm::Vec3, max: &glm::Vec3) -> Self {
        AABB {
            min: *min,
            max: *max,
        }
    }

    pub fn intersect(&self, r: &Ray) -> bool {
        for (&ro, &rd, &minc, &maxc) in izip!(
            r.o.as_slice().iter(),
            r.dir.as_slice().iter(),
            self.min.as_slice().iter(),
            self.max.as_slice().iter()
        ) {
            let t0 = (((minc - ro) / rd) as f32).min((maxc - ro) / rd);
            let t1 = (((minc - ro) / rd) as f32).max((maxc - ro) / rd);
            let t_min = TMIN.max(t0);
            let t_max = TMAX.min(t1);

            if t_max <= t_min {
                return false;
            };
        }

        true
    }

    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: glm::min2(&self.min, &other.min),
            max: glm::max2(&self.max, &other.max),
        }
    }
}
#[derive(Clone)]
pub struct HittableList {
    content: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        HittableList {
            content: Vec::new(),
        }
    }

    pub fn add(&mut self, h: Box<dyn Hittable>) {
        self.content.push(h);
    }
}

impl Hittable for HittableList {
    fn intersect(&self, r: &Ray, h: &mut Hit) -> bool {
        let mut is_hit = false;
        for hittable in &self.content {
            is_hit = hittable.intersect(r, h) || is_hit;
        }

        is_hit
    }

    fn bounding_box(&self) -> Option<AABB> {
        let mut aabb: Option<AABB> = None;
        for hittable in &self.content {
            match hittable.bounding_box() {
                Some(bb) => match aabb {
                    Some(bb0) => aabb = Some(bb0.union(&bb)),
                    None => aabb = Some(bb),
                },
                None => (),
            }
        }

        aabb
    }

    fn box_clone(&self) -> Box<dyn Hittable> {
        let copied = self.clone();
        Box::new(copied)
    }
}

#[derive(Clone)]
pub struct BVH {
    left: Option<Box<dyn Hittable>>,
    right: Option<Box<dyn Hittable>>,
    aabb: AABB,
}

impl BVH {
    pub fn new() -> Self {
        BVH {
            left: None,
            right: None,
            aabb: AABB::new(),
        }
    }

    pub fn from_hittable_list(hittables: &HittableList, rng: &mut ThreadRng) -> Self {
        Self::from_list_of_hittables(hittables.content.clone(), rng)
    }

    fn comp_x(a: &Box<dyn Hittable>, b: &Box<dyn Hittable>) -> Ordering {
        match (a.bounding_box(), b.bounding_box()) {
            (None, None) => Ordering::Equal,
            (None, _) => Ordering::Less,
            (_, None) => Ordering::Greater,
            (Some(bb0), Some(bb1)) => bb0.min.x.partial_cmp(&bb1.min.x).unwrap(),
        }
    }
    fn comp_y(a: &Box<dyn Hittable>, b: &Box<dyn Hittable>) -> Ordering {
        match (a.bounding_box(), b.bounding_box()) {
            (None, None) => Ordering::Equal,
            (None, _) => Ordering::Less,
            (_, None) => Ordering::Greater,
            (Some(bb0), Some(bb1)) => bb0.min.y.partial_cmp(&bb1.min.y).unwrap(),
        }
    }
    fn comp_z(a: &Box<dyn Hittable>, b: &Box<dyn Hittable>) -> Ordering {
        match (a.bounding_box(), b.bounding_box()) {
            (None, None) => Ordering::Equal,
            (None, _) => Ordering::Less,
            (_, None) => Ordering::Greater,
            (Some(bb0), Some(bb1)) => bb0.min.z.partial_cmp(&bb1.min.z).unwrap(),
        }
    }

    pub fn from_list_of_hittables(list: Vec<Box<dyn Hittable>>, rng: &mut ThreadRng) -> Self {
        let axis = rng.gen_range(0, 3);
        let mut objects = list;

        match objects.len() {
            1 => {
                let left = objects[0].box_clone();

                return BVH {
                    left: Some(left),
                    right: None,
                    aabb: objects.first().unwrap().bounding_box().unwrap(),
                };
            }
            2 => {
                let left = objects[0].box_clone();
                let right = objects[1].box_clone();
                let aabb = AABB::union(
                    &left.bounding_box().unwrap(),
                    &right.bounding_box().unwrap(),
                );
                return BVH {
                    left: Some(left),
                    right: Some(right),
                    aabb,
                };
            }
            _ => (),
        };

        let comp = match axis {
            0 => BVH::comp_x,
            1 => BVH::comp_y,
            _ => BVH::comp_z,
        };

        objects.sort_by(comp);

        let right_vec = objects.split_off(objects.len() / 2);
        // split_off mutated objects so it holds the first half
        let left = BVH::from_list_of_hittables(objects, rng);
        let right = BVH::from_list_of_hittables(right_vec, rng);

        let aabb = AABB::union(&left.aabb, &right.aabb);

        BVH {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            aabb,
        }
    }
}

impl Hittable for BVH {
    fn intersect(&self, r: &Ray, h: &mut Hit) -> bool {
        let mut is_hit = false;

        if self.aabb.intersect(r) {
            match &self.left {
                None => (),
                Some(hittable) => is_hit = hittable.intersect(r, h) || is_hit,
            };
            match &self.right {
                None => (),
                Some(hittable) => is_hit = hittable.intersect(r, h) || is_hit,
            }
        };

        return is_hit;
    }

    fn bounding_box(&self) -> Option<AABB> {
        Some(self.aabb.clone())
    }

    fn box_clone(&self) -> Box<dyn Hittable> {
        Box::new(self.clone())
    }
}

unsafe impl Sync for BVH {}

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

        let o = glm::vec3(4., 2.5, -7.);
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

fn raycast(r: &Ray, h: &mut Hit, scene: &dyn Hittable) -> bool {
    scene.intersect(r, h)
}

fn shade_diffuse(r: &mut Ray, h: &Hit, mat: DiffuseMat, rng: &mut ThreadRng) -> glm::Vec3 {
    r.dir = cosine_weighted_sample(&h.n, rng);
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

fn ray_color(r: &mut Ray, scene: &dyn Hittable, rng: &mut ThreadRng) -> glm::Vec3 {
    let mut h: Hit = Hit::new();
    let mut c = glm::vec3(1., 1., 1.);

    for _ in 0..MAX_ITER {
        h.t = TMAX;
        let is_hit = raycast(&r, &mut h, scene);
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

pub fn raytrace(uv: glm::Vec2, cam: &Camera, scene: &BVH, rng: &mut ThreadRng) -> glm::Vec3 {
    let mut r = cam.get_ray(uv, rng);

    ray_color(&mut r, scene, rng)
}
