use criterion::{criterion_group, criterion_main, Criterion};
use sahi::backend::{Backend, CpuBackend, CpuBackendConfig};
use sahi::detection::{BoundingBox, Detection};
use sahi::inference::{callback, ImageData};
use sahi::slicer::{Slicer, SlicerConfig};

/// Create a synthetic image of given dimensions.
fn synthetic_image(width: u32, height: u32) -> ImageData {
    let size = (width * height * 3) as usize;
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    ImageData::from_rgb(data, width, height)
}

/// Create slices for a given image size.
fn make_slices(width: u32, height: u32) -> Vec<sahi::slicer::Slice> {
    let config = SlicerConfig {
        slice_width: 640,
        slice_height: 640,
        overlap_width_ratio: 0.2,
        overlap_height_ratio: 0.2,
    };
    let slicer = Slicer::new(config);
    slicer.slice(width, height)
}

fn bench_cpu_sequential(c: &mut Criterion) {
    let image = synthetic_image(1920, 1080);
    let slices = make_slices(1920, 1080);
    let config = CpuBackendConfig {
        num_threads: 1,
        parallel_inference: false,
    };
    let backend = CpuBackend::with_config(config);

    let cb = callback(|_img: &ImageData| {
        Ok(vec![Detection::new(
            BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            0,
            0.9,
            None,
        )])
    });

    c.bench_function("cpu_sequential_1920x1080", |b| {
        b.iter(|| backend.process_slices(&image, &slices, &cb).unwrap())
    });
}

#[cfg(feature = "parallel")]
fn bench_cpu_parallel_extraction(c: &mut Criterion) {
    let image = synthetic_image(1920, 1080);
    let slices = make_slices(1920, 1080);
    let backend = CpuBackend::with_config(CpuBackendConfig {
        num_threads: 0, // auto
        parallel_inference: false,
    });

    let cb = callback(|_img: &ImageData| {
        Ok(vec![Detection::new(
            BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            0,
            0.9,
            None,
        )])
    });

    c.bench_function("cpu_parallel_extract_1920x1080", |b| {
        b.iter(|| backend.process_slices(&image, &slices, &cb).unwrap())
    });
}

#[cfg(feature = "parallel")]
fn bench_cpu_parallel_full(c: &mut Criterion) {
    let image = synthetic_image(1920, 1080);
    let slices = make_slices(1920, 1080);
    let backend = CpuBackend::with_config(CpuBackendConfig {
        num_threads: 0,
        parallel_inference: true,
    });

    let cb = callback(|_img: &ImageData| {
        Ok(vec![Detection::new(
            BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            0,
            0.9,
            None,
        )])
    });

    c.bench_function("cpu_parallel_full_1920x1080", |b| {
        b.iter(|| backend.process_slices(&image, &slices, &cb).unwrap())
    });
}

fn bench_cpu_large_image(c: &mut Criterion) {
    let image = synthetic_image(3840, 2160);
    let slices = make_slices(3840, 2160);
    let backend = CpuBackend::new();

    let cb = callback(|_img: &ImageData| {
        Ok(vec![Detection::new(
            BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            0,
            0.9,
            None,
        )])
    });

    c.bench_function("cpu_default_3840x2160", |b| {
        b.iter(|| backend.process_slices(&image, &slices, &cb).unwrap())
    });
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_cpu_sequential,
    bench_cpu_parallel_extraction,
    bench_cpu_parallel_full,
    bench_cpu_large_image,
);

#[cfg(not(feature = "parallel"))]
criterion_group!(benches, bench_cpu_sequential, bench_cpu_large_image,);

criterion_main!(benches);
