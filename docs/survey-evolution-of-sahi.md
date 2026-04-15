# Survey: The Evolution of Slicing-Aided Hyper Inference (SAHI) and Small Object Detection

> A comprehensive survey tracing the algorithmic lineage from classical sliding windows to modern adaptive tiling approaches.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Era 1: Classical Foundations (Pre-2015)](#2-era-1-classical-foundations-pre-2015)
3. [Era 2: Deep Learning Revolution (2014--2017)](#3-era-2-deep-learning-revolution-2014-2017)
4. [Era 3: Small Object Detection Focus (2017--2021)](#4-era-3-small-object-detection-focus-2017-2021)
5. [Era 4: SAHI and Systematic Tiling (2020--2022)](#5-era-4-sahi-and-systematic-tiling-2020-2022)
6. [Era 5: Post-SAHI and Modern Approaches (2022--2025)](#6-era-5-post-sahi-and-modern-approaches-2022-2025)
7. [The Parallel Evolution of NMS](#7-the-parallel-evolution-of-nms)
8. [Benchmarks and Datasets](#8-benchmarks-and-datasets)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Implementation Landscape](#10-implementation-landscape)
11. [TensorRT and Inference Optimization](#11-tensorrt-and-inference-optimization)
12. [Edge Device Deployment](#12-edge-device-deployment)
13. [Quantization and Precision](#13-quantization-and-precision)
14. [Edge-Cloud Collaborative Architectures](#14-edge-cloud-collaborative-architectures)
15. [Industry Applications](#15-industry-applications)
16. [Open Problems and Future Directions](#16-open-problems-and-future-directions)
17. [References](#17-references)

---

## 1. Introduction

Modern object detectors achieve remarkable accuracy on large, prominent objects but consistently fail on small objects -- those occupying less than 32x32 pixels in standard benchmarks. This failure is structural: detectors resize inputs to fixed resolutions (typically 640x640), causing small objects to shrink to sub-pixel representations that carry insufficient information for detection.

**Slicing-Aided Hyper Inference (SAHI)** [Akyon et al., 2022] addresses this by partitioning high-resolution images into overlapping tiles, running detection independently on each tile, remapping coordinates back to the original image space, and merging duplicate detections. The approach is model-agnostic, requires no retraining, and has become the de facto standard for small object detection in aerial, satellite, surveillance, and medical imaging.

This survey traces the complete algorithmic lineage -- from the classical sliding window paradigm through deep learning's multi-scale architectures, the emergence of scale-normalization methods, the crystallization of systematic tiling in SAHI, and the latest adaptive and content-aware approaches.

---

## 2. Era 1: Classical Foundations (Pre-2015)

### 2.1 The Sliding Window Paradigm

The conceptual ancestor of SAHI is the sliding window detector, which exhaustively scans an image at all positions and scales.

**Viola-Jones Framework** [Viola & Jones, CVPR 2001]
- Introduced the integral image for rapid Haar-like feature computation, AdaBoost-based feature selection, and a cascade classifier architecture.
- Scanned all possible locations and scales -- 15x faster than competing methods (Rowley-Baluja-Kanade).
- *SAHI connection:* Established the paradigm of scanning sub-regions at multiple scales. SAHI replaces exhaustive scanning with systematic tiling at a single scale matched to the detector's training resolution.

**Histograms of Oriented Gradients (HOG)** [Dalal & Triggs, CVPR 2005]
- Dense grids of gradient orientation histograms (8x8 pixel cells, 16x16 blocks, 9 orientation bins) paired with a linear SVM.
- 35,000+ citations -- one of the most influential feature descriptors in computer vision history.
- *SAHI connection:* HOG+SVM with sliding windows operated on fixed-size detection windows applied at multiple scales via image pyramids. This multi-scale scanning philosophy evolved into the systematic slicing in SAHI.

**Deformable Part Models (DPM)** [Felzenszwalb et al., CVPR 2008 / TPAMI 2010]
- Extended HOG with mixture models of multi-scale deformable parts trained via latent SVM.
- Divided objects into parts (e.g., a car into windows, body, wheels) with learned spatial relationships.
- *SAHI connection:* DPM's multi-scale part detection established that objects at different scales require different representational treatment -- a core insight behind SAHI's scale-normalization through slicing.

### 2.2 Image Pyramids

Image pyramids construct scaled versions of an input image (Gaussian smoothing + subsampling), running the detector at each level. This was the dominant multi-scale strategy through the pre-deep-learning era. The technique is compute- and memory-intensive, motivating both FPN (leveraging built-in multi-scale CNN features) and SAHI (slicing as an alternative to pyramids).

### 2.3 Region Proposals

**Selective Search** [Uijlings et al., IJCV 2013]
- Combined bottom-up segmentation with greedy hierarchical grouping to generate ~10,000 object proposals with 99% recall.
- *SAHI connection:* Region proposals replaced brute-force sliding windows with data-driven candidate generation -- a conceptual shift toward intelligently partitioning images rather than exhaustive scanning.

---

## 3. Era 2: Deep Learning Revolution (2014--2017)

### 3.1 The R-CNN Family

| Method | Year | Venue | Key Contribution | Speed |
|--------|------|-------|------------------|-------|
| R-CNN | 2014 | CVPR | CNN features + selective search proposals | ~49s/image |
| Fast R-CNN | 2015 | ICCV | RoI pooling on shared feature maps | ~2.3s/image |
| Faster R-CNN | 2015 | NeurIPS | Region Proposal Network (RPN) with multi-scale anchors | ~0.2s/image |

**Faster R-CNN's** multi-scale anchor strategy (3 scales x 3 aspect ratios = 9 anchors per position) partially addressed scale variation but remained insufficient for very small objects in high-resolution images. RoI pooling's concept of extracting features from specific image sub-regions parallels SAHI's slicing -- both focus computation on localized areas.

### 3.2 Single-Shot Detectors

**SSD** [Liu et al., ECCV 2016]
- Eliminated the proposal stage entirely, combining predictions from multiple feature maps at different resolutions.
- Improved mAP from 63.4% (YOLOv1) to 74.3%, but small object detection remained a core weakness because lower-resolution feature maps carry insufficient semantic information.

**YOLO and the Small Object Problem**

| Version | Year | Venue | Small Object Handling |
|---------|------|-------|-----------------------|
| YOLOv1 | 2016 | CVPR | Coarse 7x7 grid, 2 boxes/cell -- severe small object limitation |
| YOLOv2 | 2017 | CVPR | Multi-scale training (320-608px), anchor boxes, passthrough layer |
| YOLOv3 | 2018 | arXiv | Multi-scale predictions at 3 FPN levels |

YOLOv1's documented weakness -- groups of small, nearby objects were particularly problematic -- was a key motivator for the entire line of work culminating in SAHI. Each YOLO version improved but never fully resolved the small object gap.

### 3.3 Feature Pyramid Networks -- The Pivotal Paper

**FPN** [Lin et al., CVPR 2017]
- Top-down architecture with lateral connections building high-level semantic feature maps at all scales with marginal extra cost.
- Surpassed all entries from the COCO 2016 challenge as a single model.
- *SAHI connection:* FPN became the de facto backbone for multi-scale detection and is the architectural foundation most SAHI-wrapped detectors are built upon. However, even FPN's highest-resolution level has been significantly downsampled from the input. **SAHI addresses this residual gap** by ensuring small objects are represented at sufficient pixel resolution *before* entering the FPN pipeline.

### 3.4 Deformable Convolutions

**DCN v1** [Dai et al., ICCV 2017] / **DCN v2** [Zhu et al., CVPR 2019]
- Learned spatial offsets for adaptive geometric transformation, enabling receptive fields to adapt to object shape and scale.
- DCN v3 (used in InternImage, 2023) later pushed this to foundation model scale.
- *SAHI connection:* Deformable convolutions handle geometric variations adaptively at the network level. SAHI operates at the input level. The approaches are complementary.

---

## 4. Era 3: Small Object Detection Focus (2017--2021)

### 4.1 Scale Normalization Approaches

**SNIP: Scale Normalization for Image Pyramids** [Singh & Davis, CVPR 2018]
- Key insight: CNNs are not robust to scale changes. Selectively back-propagated gradients of objects only at appropriate scales within an image pyramid.
- Achieved 45.7% AP (single model) on COCO; won Best Student Entry at COCO 2017 Challenge.
- *SAHI connection:* SNIP's insight that detectors should see objects at the scale they were trained on is the **same fundamental insight** behind SAHI's slicing. By cutting large images into patches, small objects appear at "normal" training scale.

**SNIPER: Efficient Multi-Scale Training** [Singh et al., NeurIPS 2018]
- Processed 512x512 "chips" around ground-truth clusters instead of full image pyramids.
- Processed only ~30% more pixels than single-scale training while observing extreme pyramid resolutions.
- *SAHI connection:* SNIPER's chip extraction is a **direct precursor** to SAHI's slicing. The key difference: SNIPER uses chips during training with ground-truth guidance; SAHI applies systematic slicing at inference without requiring ground-truth.

**TridentNet** [Li et al., ICCV 2019]
- Parallel multi-branch architecture with shared parameters but different dilation rates for scale-specific feature maps.
- *SAHI connection:* TridentNet modifies the network architecture; SAHI modifies the input. Both address the same scale mismatch problem from different angles.

### 4.2 GAN-Based Approaches

**Perceptual GAN** [Li et al., CVPR 2017]
- Generator transfers poor small-object representations to super-resolved ones resembling large objects.

**SOD-MTGAN** [Zhang et al., ECCV 2018]
- End-to-end multi-task GAN with super-resolution network optimized jointly for detection.

*SAHI connection:* These approaches synthesize better features internally, while SAHI ensures objects are sufficiently large in pixels by slicing. SAHI requires no architectural modification or retraining -- a decisive practical advantage.

### 4.3 Clustered and Density-Aware Detection

**ClusDet** [Yang et al., ICCV 2019]
- Unified end-to-end framework: cluster proposal network (CPNet) -> scale estimation (ScaleNet) -> detection (DetecNet).
- Generated "chips" centered on object clusters rather than uniform grids.
- *SAHI connection:* ClusDet's cluster-driven chip generation is conceptually related to the adaptive slicing extensions that followed SAHI. Where SAHI tiles uniformly, ClusDet allocates chips based on predicted object density.

**QueryDet** [Yang et al., CVPR 2022 (Oral)]
- Two-step: coarse location prediction on low-resolution features, then sparse high-resolution refinement.
- +2.0 AP_small on COCO, 3x high-resolution inference speedup.
- *SAHI connection:* QueryDet sparsifies feature computation where SAHI tiles the input. The approaches are complementary -- one could use SAHI tiling with QueryDet's sparse feature extraction.

### 4.4 Competition-Driven Tiling (Pre-SAHI)

Before SAHI formalized tiling, it was widespread but ad-hoc:

**xView Challenge (2018):**
- 1st place used 700x700 crops with 80px overlap, expanding 846 images to 63,535 tiles.
- Overlapping chipping provided a **15% accuracy increase** over non-overlapping.
- Every competitive solution required tiling; no team operated on full-resolution images.

**VisDrone Challenge Series (2018--2021):**
- 2018: Scale Adaptive Image Cropping (SAIC) demonstrated crop-based detection.
- 2019: "Augmented Chip Mining" with class-balanced patch augmentation.
- 2020-2021: Tiling became standard infrastructure; top solutions separated by <1 AP point.

**SIMRDWN** [Van Etten, WACV 2019]
- Unified sliding-window pipeline for satellite imagery combining YOLT with TF Object Detection API.
- Processed at >0.2 km^2/s at native resolution.
- *SAHI connection:* SIMRDWN demonstrated the necessity of windowed detection for satellite imagery but lacked SAHI's systematic overlap handling, NMM post-processing, and framework-agnostic design.

**The fundamental problem:** Every team reinvented the wheel with incompatible tiling strategies.

---

## 5. Era 4: SAHI and Systematic Tiling (2020--2022)

### 5.1 The SAHI Paper

**"Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"**
Fatih Cagatay Akyon, Sinan Onur Altinuc, Alptekin Temizel. IEEE ICIP 2022, Bordeaux. [arXiv:2202.06934]

**What was genuinely novel versus prior tiling work:**

1. **Principled Slicing Framework.** Formalized image slicing into a reproducible framework with configurable slice dimensions, overlap ratios, and resolution parameters. Prior work was competition-specific and unreproducible.

2. **Sliced Fine-Tuning.** Novel contribution: augmenting training data with sliced crops so the detector learns slice-boundary artifacts. This closed the distribution gap between training (full images at moderate resolution) and inference (high-resolution slices).

3. **GREEDYNMM.** Greedy Non-Maximum Merging: instead of discarding overlapping boxes (NMS) or re-scoring them (Soft-NMS), GREEDYNMM *merges* overlapping predictions via weighted bounding box combination. Uses Shapely STRtree for efficient spatial indexing. Supports both IoU and IoS (Intersection over Smaller area) metrics.

4. **Full-Image + Sliced Fusion.** Dual-path approach: detection on both the full image (large objects) and all slices (small objects), with merged results.

5. **Framework-Agnostic Design.** First tiling library supporting any object detection model via a uniform `AutoDetectionModel` interface.

### 5.2 The Algorithm

```
Input: Image I (W x H), detector D, slice size (sw, sh), overlap ratio r

1. SLICE: Compute grid with step = sw * (1 - r)
   For 2560x1920, sw=640, r=0.2: step=512, grid ~5x4 = 20 slices

2. EXTRACT: Crop each slice from I (pure memory copy)

3. INFER: Run D on each slice independently
   Detections are in slice-local coordinates

4. REMAP: Translate each detection by (slice.x, slice.y)
   Detections now in full-image coordinates

5. (Optional) FULL-IMAGE: Run D on resized full image
   Add detections to the pool

6. FILTER: Remove detections below confidence threshold

7. POSTPROCESS: Apply NMS/NMM/GREEDYNMM to remove duplicates
   from overlapping regions

Output: Merged detections in full-image coordinates
```

### 5.3 Experimental Results

| Detector | Dataset | Baseline AP50 | +SAHI AP50 | +SF+SAHI AP50 | Gain |
|----------|---------|---------------|------------|---------------|------|
| FCOS | VisDrone | 29.8% | 36.5% | 42.5% | +12.7% |
| VFNet | VisDrone | 31.9% | 37.0% | 45.3% | +13.4% |
| TOOD | VisDrone | 29.4% | 34.7% | 43.9% | +14.5% |
| FCOS | xView | 2.2% | 8.9% | 14.9% | +12.7% |
| TOOD | xView | 2.1% | 7.4% | 16.6% | +14.5% |

On xView, baseline AP_small was effectively **zero** (0.1% for FCOS). SAHI brought it to 12.2% -- an order-of-magnitude improvement demonstrating that standard detectors are fundamentally broken on satellite imagery without tiling.

### 5.4 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slice_width/height` | 640 | Tile size (should match detector's input resolution) |
| `overlap_width/height_ratio` | 0.2 | Fraction of overlap between tiles (0.0--0.9) |
| `confidence_threshold` | 0.25 | Minimum detection score |
| `match_threshold` | 0.5 | IoU/IoS above which boxes are considered duplicates |
| `match_metric` | IOU | IOU (standard) or IOS (better for nested boxes) |
| `class_aware` | true | Only compare boxes of the same class |
| `postprocess_type` | GREEDYNMM | NMS / NMM / GREEDYNMM |
| `include_full_image` | false | Also run on full resized image |

---

## 6. Era 5: Post-SAHI and Modern Approaches (2022--2025)

### 6.1 Adaptive and Content-Aware Slicing

The primary limitation of SAHI is uniform slicing -- every region gets equal treatment regardless of content. Post-SAHI research focuses on intelligent slice allocation.

**ASAHI: Adaptive Slicing-Aided Hyper Inference** [Remote Sensing, 2023]
- Adaptively generates 6 or 12 patches based on image dimensions using a differentiation threshold.
- Eliminates redundant computation for varying-resolution images.
- Combined with TPH-YOLOv5 for enhanced small-object sensitivity.

**GOIS: Guided Object Inference Slicing** [Neurocomputing, 2025]
- Two-stage coarse-to-fine slicing:
  - Coarse stage: 640px slices with broad coverage
  - Fine stage: 256px slices concentrated on high-density regions
  - Dynamic overlap adjusted based on object density
- Results on VisDrone2019: mAP 0.12 -> 0.33 (+175% on YOLO11), AP_small +278%, AR_small +279%.

**Dynamic Tiling** [arXiv:2309.11069, 2023]
- Starts with non-overlapping tiles for initial detections.
- Uses dynamic overlapping rates and a tile minimizer to reduce computation.
- Large-small filtering mechanism for multi-scale detection.

**Scene Heatmap-Guided Adaptive Tiling** [MDPI Symmetry, 2025]
- Attention-guided dual-resolution processing:
  - High-attention regions: fine 640x640 tiles with 20% overlap, full model
  - Low-attention regions: coarse 1024x1024 tiles, lightweight model
  - Heatmap-driven computational resource allocation.

**Altitude-Aware Dynamic Tiling** [2025]
- Scales and adaptively subdivides images based on UAV altitude.
- Reduces computation while maintaining performance for maritime small object detection.

### 6.2 Transformer-Based Small Object Detection

**DINO** [Zhang et al., ICLR 2023]
- Contrastive denoising training, look-forward-twice prediction, mixed query selection.
- 49.4 AP in 12 epochs / 63.3 AP with SwinL + Objects365 pretraining.
- NMS-free via set-based Hungarian matching.

**Grounding DINO** [Liu et al., ECCV 2024]
- Open-vocabulary detection: 52.5 zero-shot AP on COCO without COCO training data.
- *Grounding DINO + SAHI* enables open-vocabulary small object detection in high-resolution imagery -- detecting text-described objects that may be tiny in the image.

**RT-DETR** [Zhao et al., CVPR 2024]
- First real-time end-to-end detector: 53.1% AP at 108 FPS on T4.
- Flexible speed tuning via decoder layer count.
- Efficiency makes it a strong candidate for SAHI-enhanced pipelines where many slices must be processed quickly.

**RF-DETR** [Roboflow, submitted ICLR 2026]
- DINOv2 backbone + weight-sharing NAS for accuracy-latency Pareto curves.
- First real-time model to exceed 60 AP on COCO. Particularly strong on small objects.

**InternImage** [Wang et al., CVPR 2023 Highlight]
- DCNv3 as core operator, 1B+ parameters, 400M training images.
- 65.4% box mAP on COCO -- first model exceeding 65.0.
- For deployment on high-resolution imagery with tiny objects, SAHI-style slicing remains complementary even with such powerful backbones.

### 6.3 YOLO Evolution for Small Objects

| Version | Year | Key Innovation for Small Objects | AP |
|---------|------|----------------------------------|-----|
| YOLOv8 | 2023 | C2f module (improved gradient flow) | 53.9 |
| YOLOv9 | 2024 | Programmable Gradient Information (PGI) | 55.6 |
| YOLOv10 | 2024 | NMS-free training, advanced loss functions | 54.4 |
| YOLOv11 | 2024 | C3K2 module, C2PSA spatial attention | 54.7 |
| YOLOv12 | 2025 | Area Attention, R-ELAN, FlashAttention | 55.2 |
| YOLO26 | 2025 | STAL (small-target-aware label assignment), natively NMS-free | -- |

YOLO26 (September 2025, Ultralytics) introduces **STAL** (Small-Target-Aware Label Assignment) specifically for small objects, plus ProgLoss for progressive loss balancing. 43% faster on CPUs. Natively end-to-end (no NMS).

Despite continuous architectural improvements, **SAHI remains complementary to all YOLO versions** because the core problem -- resizing high-resolution inputs to 640x640 destroys small object information -- is not addressable by architecture alone.

### 6.4 Focus-and-Detect / Zoom-In Approaches

**Focus-and-Detect** [Signal Processing: Image Communication, 2022]
- Two-stage: GMM-supervised detector generates focused regions; second stage detects within focal regions.

**AdaZoom** [arXiv:2106.10409, 2021]
- RL-based focus region generation with policy gradient. Reward formulated by object distributions.

**Adaptive Image Zoom-In** [arXiv:2602.07512, 2025]
- Non-uniform zoom-in to magnify objects of interest.

*SAHI connection:* These represent content-aware alternatives to uniform slicing. Where SAHI tiles uniformly, zoom approaches adaptively allocate resolution. They are potentially more efficient but require additional model components and training.

### 6.5 Edge and Optimized Deployment

**EdgeDuet** [Wang et al., IEEE INFOCOM 2021]
- First edge-device collaborative framework: large objects detected locally on mobile, small objects offloaded to edge server via tiling.
- RoI frame encoding reduces transmitted data; tile-level parallelism for pipelined offloading.

**TensorRT + SAHI on Jetson** [HouYanSong/tensorrtx-yolov8-sahi]
- YOLOv8 Int8 TensorRT on Jetson Orin Nano (8GB): slice + batch inference in 0.04s; 1080p video with SAHI + ByteTrack at ~15 FPS.

---

## 7. The Parallel Evolution of NMS

Non-Maximum Suppression is the critical post-processing component that makes tiled detection viable. Its evolution directly enables SAHI's effectiveness.

### 7.1 Timeline

| Year | Method | Authors / Venue | Key Innovation |
|------|--------|----------------|----------------|
| 1971 | NMS (edge detection) | Rosenfeld & Thurston, IEEE Trans. | Original concept for edge thinning |
| ~2005 | Greedy NMS (detection) | Standard practice | Sort by score, suppress overlapping boxes above IoU threshold |
| 2017 | **Soft-NMS** | Bodla et al., ICCV | Continuous score decay instead of hard suppression; +1-2% mAP |
| 2017 | **Learned NMS** | Hosang et al., CVPR | Neural network (GossipNet) replaces hand-crafted NMS |
| 2018 | **Relation Networks** | Hu et al., CVPR | Attention-based duplicate removal with geometric weights |
| 2019 | **Adaptive NMS** | Liu et al., CVPR | Learns per-object density score as adaptive threshold |
| 2020 | **DIoU-NMS** | Zheng et al., AAAI | Considers both overlap area and center-point distance |
| 2020 | **Matrix NMS** | Wang et al. (SOLOv2), NeurIPS | Parallel matrix operations in one shot |
| 2021 | **Cluster-NMS** | Zheng et al., IEEE T-Cybernetics | Parallel matrix ops with geometric factors; +1.7 AP, +6.2 AR100 |
| 2021 | **Weighted Boxes Fusion** | Solovyev et al., Image & Vision Computing | Confidence-weighted average boxes; strong for ensembles |
| 2021 | **NMS-Loss** | Luo et al., ICMR | Trainable NMS via pull/push loss; no additional parameters |
| **2022** | **GREEDYNMM** | **Akyon et al., ICIP (SAHI)** | **Greedy merging via weighted bounding box combination with STRtree spatial indexing** |
| 2024-25 | **NMS-Free** | YOLOv10, YOLO26, RT-DETR, DINO | End-to-end set prediction eliminates NMS entirely |

### 7.2 SAHI's Three Postprocessing Algorithms

**NMS (Non-Maximum Suppression)**
- Sort by confidence descending. For each box, suppress all lower-confidence boxes exceeding the match threshold.
- Fast and well-understood but can discard valid adjacent detections.

**NMM (Non-Maximum Merging)**
- Groups all overlapping boxes. Merges each group into a union bounding box with averaged confidence.
- Better when tile-boundary detections cover different parts of the same object.

**GREEDYNMM (default)**
- Processes greedily in confidence order. Absorbs overlapping boxes into the current detection by taking the union of bounding boxes and accumulating confidence.
- The merged box **grows** as it absorbs, potentially pulling in more boxes iteratively.
- Uses STRtree spatial indexing (O(n log n)) rather than pairwise comparison (O(n^2)).
- Supports both IoU and IoS metrics and class-aware filtering.

### 7.3 Why GREEDYNMM Matters for Tiled Detection

Standard NMS was designed for single-image inference where duplicates are redundant firings on the same object. In tiled inference, duplicates arise because **overlapping tiles independently detect the same object, often with slightly different bounding boxes**. Hard suppression (NMS) arbitrarily picks one; merging (GREEDYNMM) combines information from both, producing a tighter final box. This is particularly important at tile boundaries where each tile may see only a partial view of the object.

---

## 8. Benchmarks and Datasets

### 8.1 Evolution of Key Datasets

| Dataset | Year | Images | Instances | Classes | Avg Object Size | Resolution | Domain |
|---------|------|--------|-----------|---------|-----------------|------------|--------|
| PASCAL VOC | 2005-12 | ~21K | ~52K | 20 | Large | ~500x375 | Consumer photos |
| MS COCO | 2014 | 330K | 1.5M | 80 | Mixed (41% small) | ~640x480 | Everyday scenes |
| Stanford Drone | 2016 | 60 videos | ~20K | 6 | Small-medium | Bird's-eye video | Campus |
| UAVDT | 2018 | 77,819 | 835,879 | 4 | Small-medium | 1080x540 | Vehicle tracking |
| **VisDrone** | 2018-21 | 10,209 | 2.6M | 10 | **10-30px (~65% small)** | 1920x1080 | **Drone surveillance** |
| **xView** | 2018 | 1,127 | 601K | 60 | **~13px (cars)** | Satellite (0.3m GSD) | **Satellite imagery** |
| DOTA v1.0/v2.0 | 2018/21 | 2,806/11,268 | 188K/1.79M | 15/18 | Variable | 800-20,000px | Aerial (oriented) |
| **AI-TOD** | 2020 | 28,036 | 700K | 8 | **12.8px mean** | Aerial | **Tiny aerial objects** |
| TinyPerson | 2020 | 1,610 | 72,651 | 2 | <20x20px | Wide-field | Tiny persons |
| SODA-D/SODA-A | 2023 | 24,828/2,513 | 278K/872K | 9/9 | Small | Driving/Aerial | Multi-scenario SOD |
| EV-UAV | 2025 | 147 seqs | 2.3M | -- | **6.8x5.4px** | Event camera | Anti-UAV |

**Bold entries** are the primary SAHI benchmarks and the most extreme tiny-object datasets.

### 8.2 Why VisDrone Became SAHI's Primary Benchmark

1. **Overwhelming small objects:** ~65% of instances are below COCO's 32x32 "small" threshold.
2. **High resolution:** 1920x1080+ images make 640x640 tiling directly applicable.
3. **Dense scenes:** 100+ objects per frame stress-test the NMS/merging pipeline.
4. **Multi-year challenge:** Longitudinal data showing tiling's progression from specialty to standard.

### 8.3 The Rise of Tiling in Competitions

| Year | Event | Tiling Status |
|------|-------|---------------|
| 2018 | xView Challenge | 1st place: 700x700 crops, 80px overlap (+15% accuracy) |
| 2018 | VisDrone-DET | SAIC among top methods; cropping emerging |
| 2019 | VisDrone-DET | "Augmented Chip Mining" formalized training-time tiling |
| 2020 | SAHI library released | Framework-agnostic tiling now available to everyone |
| 2020-21 | VisDrone-DET | Tiling standard in all top solutions |
| 2022 | SAHI paper (ICIP) | Formal validation across FCOS, VFNet, TOOD |
| 2023+ | General adoption | Tiling assumed infrastructure; focus shifts to *adaptive* tiling |

---

## 9. Evaluation Metrics

### 9.1 PASCAL VOC Era (2005--2012)
- **mAP@0.5**: Single IoU threshold. No size stratification. Poor small-object performance hidden by strong large-object scores.

### 9.2 MS COCO Era (2014--present)
- **AP**: Averaged over 10 IoU thresholds [0.50, 0.55, ..., 0.95]
- **AP_small**: area < 32^2 px -- **the critical metric for SAHI**
- **AP_medium**: 32^2 <= area < 96^2 px
- **AP_large**: area >= 96^2 px
- **AR_1/10/100**: Average Recall at different max-detection limits

### 9.3 Why AP_small Is the Metric That Matters

1. **Exposes the core failure:** Typical AP_large - AP_small gap is 25-30 points. On xView, AP_small can be effectively zero without tiling.
2. **Directly measures what SAHI fixes:** On VisDrone, SAHI improved TOOD's AP50_small from 18.1% to 31.7% (+75% relative).
3. **Dominates overall AP:** Because VisDrone is ~65% small objects, AP_small largely determines overall performance.

### 9.4 TIDE Error Analysis [ECCV 2020]

TIDE decomposes errors into: Classification, Localization, Both, Duplicate, Background, Missed.

For small objects, **Missed Detection** dominates -- the detector never fires. This directly validates SAHI: the problem is not misclassification or poor localization, but complete failure to detect. TIDE also reveals that tiling introduces **Duplicate Detection** errors at boundaries, highlighting the importance of SAHI's NMS/merging pipeline.

---

## 10. Implementation Landscape

### 10.1 The Python Reference: obss/sahi

| Metric | Value |
|--------|-------|
| First release | February 23, 2021 (v0.3.0) |
| Latest release | September 28, 2025 (v0.11.36) |
| Total releases | 121 |
| GitHub stars | ~5,200 |
| PyPI downloads | 6.69 million |
| Contributors | 75 |
| Citations | 400-600+ |

**Supported backends (chronological):**

| Backend | Since | Notes |
|---------|-------|-------|
| Detectron2 | v0.3.x (2021) | Launch backend |
| MMDetection | v0.3.x (2021) | Launch backend |
| YOLOv5 | v0.3.x (2021) | Launch backend |
| HuggingFace | ~v0.8.x (2022) | DETR, YOLOS, etc. |
| Ultralytics | v0.11.x (2022) | YOLOv8 through YOLO26, RT-DETR |
| TorchVision | v0.11.x (2022) | Faster/Mask R-CNN |
| Roboflow/RF-DETR | v0.11.26 (2025) | Latest addition |

### 10.2 sahi.rs (This Project)

A Rust implementation targeting high-performance deployment:

| Module | Purpose |
|--------|---------|
| `slicer.rs` | Tile grid generation with configurable overlap |
| `inference.rs` | Model-agnostic `InferenceCallback` trait |
| `postprocess.rs` | NMS / NMM / GREEDYNMM with IoU/IoS |
| `backend/cpu.rs` | Sequential + parallel (Rayon) |
| `backend/cuda.rs` | GPU slice extraction via custom PTX kernels + CUDA streams |
| `onnx/yolov8/` | Built-in YOLOv8 detector (ONNX Runtime) |
| `lib.rs` | PyO3 Python bindings |

Key advantages over Python SAHI:
- GPU-accelerated slice extraction (custom CUDA kernels)
- Multi-stream pipelining (extract N+1 while inferring N)
- CPU parallel extraction via Rayon
- Batch inference interface
- Zero-copy buffer reuse
- Feature-gated compilation (cuda, python, onnx, models)

### 10.3 C++/CUDA Implementations

| Project | Focus | Performance |
|---------|-------|-------------|
| trt-sahi-yolo | TensorRT + SAHI for YOLOv5/v8/v11 | CUDA kernel tiling |
| tensorrtx-yolov8-sahi | Int8 TensorRT on Jetson | 0.04s slice+infer, ~15 FPS 1080p |

### 10.4 Integration Ecosystem

- **Ultralytics:** Official documentation for SAHI integration with all YOLO models.
- **Roboflow:** Native `InferenceSlicer` in their SDK and workflow blocks.
- **FiftyOne (Voxel51):** Interactive visualization and evaluation of SAHI predictions.
- **Azure ML:** Native AutoML tiled inference for small objects (SAHI-like functionality).
- **Hugging Face:** Multiple Spaces demos; paper page; model discovery.

---

## 11. TensorRT and Inference Optimization

### 11.1 TensorRT Fundamentals

TensorRT is NVIDIA's inference optimizer that converts trained neural networks into highly optimized engines. Its core optimization techniques are directly relevant to SAHI's multi-slice pipeline:

**Layer Fusion:** Combines sequential operations (e.g., Conv-BN-ReLU) into single CUDA kernels, eliminating intermediate memory writes. Three fusion types: vertical (sequential ops), horizontal (shared-input ops), and elimination (redundant ops). On edge GPUs where memory bandwidth is the primary bottleneck (68--204 GB/s vs 1000+ GB/s on datacenter GPUs), fusion can yield 30--50% latency reduction.

**Kernel Auto-Tuning:** Profiles hundreds of kernel implementations per layer on the target GPU (GEMM, Winograd, FFT, implicit GEMM), selecting the fastest. Results are cached in the serialized engine file. On hardware with Tensor Cores, TensorRT leverages 8x FP16 and 16x INT8 throughput automatically.

**Precision Modes:**

| Mode | Speedup vs FP32 | Accuracy Impact | Use Case |
|------|-----------------|-----------------|----------|
| FP32 | 1x (baseline) | None | Development/debugging |
| FP16 | ~2x | Negligible (<0.1 mAP) | **Safe default for SAHI on edge** |
| INT8 | ~4x | Moderate (3--7 mAP) | Latency-critical deployments |
| Mixed | ~3x | Minimal | Best accuracy-speed tradeoff |

### 11.2 TensorRT and SAHI: Dynamic Batch Sizes

SAHI's slice count varies with image resolution and overlap settings. A 1080p image with 640x640 slices and 0.2 overlap produces ~8 slices; a 4K image produces ~28. TensorRT handles this via optimization profiles:

```
Profile: {
  min: (1, 3, 640, 640),    // single slice
  opt: (16, 3, 640, 640),   // typical batch
  max: (49, 3, 640, 640),   // max (7x7 grid)
}
```

**Best practice:** Pad edge slices to uniform dimensions (e.g., 640x640) rather than using dynamic spatial shapes. Dynamic shapes trigger kernel reselection with 10--100x latency spikes. The padding overhead is trivial compared to reselection penalties.

### 11.3 Execution Contexts and CUDA Streams

Each TensorRT `IExecutionContext` can be bound to a CUDA stream, but calling `enqueueV2()` from the same context on different streams concurrently produces undefined behavior. For concurrent inference across streams, one execution context per stream is required.

This maps directly to sahi.rs's multi-stream architecture: the CUDA backend distributes slice extraction across streams round-robin, with pipelining that overlaps GPU extraction of chunk N+1 with CPU inference of chunk N.

### 11.4 TensorRT + SAHI Implementations

**trt-sahi-yolo** [leon0514/trt-sahi-yolo] -- C++ (81.8%), CUDA (16.5%), 97 stars.
- Supports YOLOv5/v8/v11, D-FINE, YOLOE with TensorRT 8 and 10.
- CUDA-based slicing with configurable overlap. All slices batched into single inference call.
- Coordinate remapping via CUDA decode kernels receiving per-slice offsets.
- Global NMS on GPU across all slice detections simultaneously.
- Constraint: total slices must not exceed `max_batch_size` set at engine build time.

**tensorrtx-yolov8-sahi** [HouYanSong] -- YOLOv8s + INT8 on Jetson Orin Nano (8GB).
- Slice + batch inference: **0.04s** per frame.
- 1080p video with SAHI + ByteTrack: **~15 FPS**.
- Fixed batch size 8, max input 3000x3000, max 1000 output boxes.

### 11.5 ONNX Runtime vs Native TensorRT

| Aspect | Native TensorRT | ONNX Runtime TRT-EP | ONNX Runtime CUDA-EP |
|--------|----------------|---------------------|---------------------|
| Latency | Lowest | ~5--15% overhead | ~20--40% overhead |
| Portability | NVIDIA only | Cross-vendor | Cross-vendor |
| INT8 support | Full (PTQ + QAT) | Via calibration | Limited |
| Rust integration | C API bindings | **ort crate** (pyke/ort) | **ort crate** |
| Dynamic shapes | Full | Supported | Full |

The **ort** crate (v2.0.0-rc.12) provides Rust bindings for ONNX Runtime with TensorRT EP. Enable via the `tensorrt` Cargo feature. For Jetson deployment, ONNX Runtime must be built from source for aarch64 + CUDA. The `onnx-tensorrt` feature in sahi.rs connects through this path.

---

## 12. Edge Device Deployment

### 12.1 Edge Hardware Landscape

| Device | AI Performance | Memory | Power | SAHI Viability |
|--------|---------------|--------|-------|----------------|
| **Jetson Orin Nano 8GB** | 40 TOPS (INT8) | 8GB | 7--15W | Good -- **15 FPS 1080p proven** |
| **Jetson Orin NX 16GB** | 100 TOPS (INT8) | 16GB | 10--25W | Very good |
| **Jetson AGX Orin 64GB** | 275 TOPS (INT8) | 64GB | 15--60W | Excellent |
| **Hailo-8 + RPi5** | 26 TOPS | 8GB (RPi) | ~5W | Promising -- 128 FPS YOLOv8n per-slice |
| **Hailo-8L + RPi5** | 13 TOPS | 8GB (RPi) | ~2.5W | Limited but lightweight |
| **Qualcomm QCS8550** | 52 TOPS | varies | ~12W | Competitive with Orin NX |
| **Apple M4 (ANE)** | ~38 TOPS | varies | varies | Good for iOS/macOS |
| **Rockchip RK3588** | 6 TOPS | varies | ~8W | Limited -- low slice counts only |
| **Google Coral** | 4 TOPS | varies | 2W | Not recommended (abandoned) |
| Intel Movidius | 4 TOPS | varies | ~1W | **Discontinued** |

### 12.2 Jetson Orin Benchmarks with SAHI

**Per-slice inference latency (TensorRT, 640x640 input):**

| Model | Orin Nano INT8 | Orin NX FP16 | Orin NX INT8 | AGX Orin FP16 |
|-------|---------------|--------------|--------------|---------------|
| YOLOv8n | 23.2ms | 5.3ms | 4.5ms | 2.6ms |
| YOLOv8s | 28.3ms | 7.9ms | 6.1ms | ~4ms |
| YOLO11n | -- | 5.3ms | 4.5ms | -- |
| YOLO11m | -- | 15.6ms | 10.4ms | -- |

**SAHI pipeline latency estimates (batched):**

| Image | Slices | Orin Nano INT8 | Orin NX FP16 | AGX Orin FP16 |
|-------|--------|---------------|--------------|---------------|
| 1080p (1920x1080) | ~8 | ~50ms | ~47ms | ~25ms |
| 4K (3840x2160) | ~28 | ~180ms | ~163ms (seq) / ~30ms (batch) | ~20ms (batch) |

### 12.3 Jetson Memory and Power Considerations

**Unified memory:** Jetson uses shared CPU/GPU LPDDR5. CPU allocations directly reduce available GPU memory. For SAHI + TensorRT on 4K: engine (~8 MB FP16 YOLOv8n) + activation (~200--400 MB for batch=28) + detection buffers (~50--100 MB) = **~500 MB--1 GB total**.

**Power modes and SAHI throughput:**

| Mode | Power | Sustained Use Case |
|------|-------|--------------------|
| 7W | Battery/thermal constrained | Low-resolution SAHI, <5 slices |
| 15W | Standard edge deployment | 1080p SAHI, ~15 FPS |
| 25W (Super Mode) | Maximum performance | 4K SAHI, batched |

Sustained throughput above 15W requires active cooling. Thermal throttling engages at 80C SoC junction temperature.

### 12.4 Inference Framework Comparison for Edge

| Framework | Hardware | INT8 | Batched SAHI | Maturity |
|-----------|----------|------|-------------|----------|
| **TensorRT** | NVIDIA GPU | Full (PTQ + QAT) | Native batch | Excellent |
| **OpenVINO** | Intel CPU/iGPU | Full | Async API | Good |
| **CoreML** | Apple ANE/GPU | FP16 primary | Metal dispatch | Good |
| **Hailo SDK** | Hailo-8/8L | INT8 native | Manual | Fair |
| **SNPE/QNN** | Qualcomm NPU | INT8 native | Manual | Fair |
| **TFLite** | ARM/Coral | INT8 via calibration | Manual | Good |
| **ort (Rust)** | Cross-platform | Via TRT-EP | Full | Good |

**DeepStream + SAHI (2025):** Native SAHI plugins for DeepStream 8.0 with TensorRT 10 showed +222% to +3600% detection count improvement for crowded pedestrians. Pipeline: `nvstreammux -> nvsahipreprocess -> nvinfer -> nvsahipostprocess -> sink`.

### 12.5 Rust on Edge Devices

**Advantages over C++/Python:**
- No garbage collector -- deterministic sub-millisecond latency
- Compile-time memory safety -- 98% of multi-threaded Rust apps showed zero race conditions vs 35% for C++
- OpenAtom benchmark: 35% faster inference and 20% less memory than C++ equivalents on embedded RTOS

**Cross-compilation for Jetson (aarch64):**
```bash
cross build --target aarch64-unknown-linux-gnu --release
```

**Crate compatibility:**
- **cudarc** (0.19.4): Supports aarch64-unknown-linux-gnu natively. Dynamic loading mode works with JetPack-installed CUDA libraries.
- **ort** (2.0.0-rc.12): No prebuilt aarch64+CUDA binary -- must build ONNX Runtime from source for Jetson GPU acceleration. TensorRT EP stabilized in ort 1.15.5.
- **Tract**: Pure-Rust ONNX inference, 100--500ms on RPi. Good for CPU-only edge.

---

## 13. Quantization and Precision

### 13.1 Impact on Small Object Detection

Small objects are **disproportionately affected by quantization**. Their feature representations have lower signal-to-noise ratios, so INT8 rounding errors destroy more discriminative information proportionally.

| Precision | mAP50-95 (YOLO26n) | Inference (ms) | Engine Size |
|-----------|---------------------|----------------|-------------|
| FP32 | 0.477 | 7.01ms | 11.4 MB |
| FP16 | 0.479 | 4.13ms | 8.0 MB |
| INT8 | 0.449 | 3.49ms | 5.5 MB |

FP16 preserves accuracy perfectly (within 0.002 mAP) while cutting latency 41%. INT8 saves an additional 15% latency but costs ~3 mAP points. For a 20-slice SAHI pipeline, FP16 saves ~58ms total vs FP32; INT8 saves ~70ms.

**Recommendation:** FP16 is the safe default for SAHI small-object detection. INT8 only when latency budget demands it.

### 13.2 Calibration for SAHI Pipelines

**Calibration dataset:** Use sliced crops matching inference dimensions. If SAHI processes 640x640 crops, calibrate with 640x640 images. Using full-resolution images creates a distribution mismatch that degrades INT8 accuracy.

**Sample count:** NVIDIA recommends 500+ images for acceptable accuracy, 1000+ for better results. Ultralytics uses MinMax calibration (empirically validated for YOLO).

**Calibration algorithms:**

| Algorithm | Mechanism | Best For |
|-----------|-----------|----------|
| MinMax | Largest absolute value as threshold | Fast prototyping (Ultralytics default) |
| Entropy (KL-Div) | Minimizes FP32/INT8 distribution divergence | Precision-critical |
| MSE | Minimizes mean-squared error | Regression-heavy tasks |
| Percentile | Threshold at 99.9th percentile | Data with outliers |

### 13.3 Mixed Precision

TensorRT supports per-layer precision. For YOLO detectors, the most quantization-sensitive layers are:
- Detection head regression branches (bounding box outputs)
- Sigmoid/Softmax layers
- First and last convolutional layers

Setting both INT8 and FP16 flags lets TensorRT automatically fall back to FP16 for sensitive layers.

### 13.4 QAT vs PTQ

| Approach | Accuracy | Effort | Use Case |
|----------|----------|--------|----------|
| PTQ (Post-Training) | 25.2 mAP (YOLO-X Tiny) | No retraining | First attempt |
| QAT (Quantization-Aware) | 30.3 mAP (YOLO-X Tiny) | 10--20 epochs fine-tuning | When PTQ drops >2 mAP |

QAT integrates fake quantization during training, letting the model learn to compensate. The gap is larger for detection than classification.

### 13.5 NMS Optimization on Edge

**GPU NMS approaches relevant to SAHI:**

- **Work-Efficient Parallel NMS** [Oro et al.]: Map/reduce with boolean adjacency matrix. Near-linear time. Tegra X1: 7.36ms for ~3000 detections.
- **NMS-Raster** [Fluendo, 2025]: Reframes NMS as GPU rendering. Bounding boxes as 2D quads with confidence as Z-coordinate. GPU depth testing auto-suppresses lower-confidence overlaps. **Linear scalability**, sub-millisecond for 4000 boxes on RTX 3060.
- **NMS-Free detection** (YOLO26, RT-DETR): Eliminates NMS entirely via end-to-end set prediction. 43% faster CPU inference. However, tiled inference still produces duplicates from overlapping tiles.

---

## 14. Edge-Cloud Collaborative Architectures

### 14.1 EdgeDuet

[Wang et al., IEEE INFOCOM 2021 / IEEE/ACM ToN 2022]

The first edge-device collaborative framework for tiling-based small object detection:

- **Mobile (local):** Lightweight detector on downsampled frames for large objects (cheap, no bandwidth cost).
- **Edge server (offloaded):** Receives selected high-resolution tiles for small objects (only tiles likely containing small objects are sent).
- **Tile selection:** Regional feature prediction scores tiles by small-object likelihood. Only high-priority tiles are offloaded.
- **Results:** +233% small object accuracy, +44.7% overall accuracy, -34.2% latency over prior offloading schemes.

### 14.2 Split Computing for SAHI

| Stage | Location | Rationale |
|-------|----------|-----------|
| Slicing | Device/Edge | Lightweight, no model needed |
| Tile filtering | Device/Edge | Reduces bandwidth |
| Inference | Edge server / Cloud | GPU-intensive |
| NMS/Merging | Edge or Device | Relatively lightweight |
| Full-image inference | Edge server | Optional, for large objects |

### 14.3 Bandwidth Optimization

- **Selective tile offloading** (EdgeDuet): Only high-priority tiles transmitted
- **Feature compression** (EC5): Information bottleneck theory compresses intermediate features
- **Non-Penetrative Tensor Partitioning (NPTP)**: Tile boundaries aligned to avoid convolution kernel overlap
- **Adaptive compression**: Quality adjusted per network conditions

---

## 15. Industry Applications

### 15.1 Drone / UAV Surveillance

**Companion computer comparison for drones:**

| Platform | TOPS | Power | Weight | TOPS/Watt |
|----------|------|-------|--------|-----------|
| Jetson Orin Nano | 34--67 | 7--15W | ~60g | 2.3--4.9 |
| Hailo-8 + RPi5 | 26 | ~5W | ~80g | ~10.4 |
| Hailo-8L | 13 | ~2.5W | ~15g | ~5.2 |

**Altitude-Adaptive SAHI (ASAHI)** [Remote Sensing, 2023]: Adaptively slices based on image dimensions rather than fixed tile sizes. +0.9% mAP50, -20--25% compute time vs standard SAHI. Directly applicable to UAVs where altitude changes object scale.

### 15.2 Maritime Surveillance

- SAR ship detection with mixed-precision TensorRT: **208 FPS** on 640x640 images (3.41x baseline).
- AI-driven tethered drone for maritime: field-tested in Trondheim, Norway (May 2025) and port of Valencia, Spain (September 2025).
- SAHI's sequential Python processing highlighted as a limitation -- motivating C++/Rust implementations like trt-sahi-yolo and sahi.rs.

### 15.3 Traffic Monitoring

SAHI + YOLOv8 deployed on Android-based Roadside Units for real-time traffic violation detection. Edge devices communicate via MQTT. For SAHI-based traffic, Orin NX or better recommended for real-time throughput.

### 15.4 Agricultural Monitoring

CCOD-Dataset (2025): 3,986 images, 410,910 annotations using SAHI + YOLOv10n for crop canopy detection from UAV orthophoto maps. SAHI accelerates inference for large-scale farmland while maintaining detection quality for small crop organs.

### 15.5 Industrial Inspection

SAHI preprocessing improves detection of small/thin defects (scratches) in precision manufacturing. YOLOv8 Nano on Raspberry Pi 500: precision 0.932, F1 0.914. TensorRT and OpenVINO enable real-time quality inspection on edge.

### 15.6 Search and Rescue

**TEXSAR** (Texas Search and Rescue) deploys SAHI with YOLO for their Automated Drone Image Analysis Tool:
- 1280x1280 or 2048x2048 slices
- YOLO trained specifically for aerial person detection
- ONNX Runtime for deployment (no external CUDA dependency)
- Processes RGB, HSV, and LAB color spaces

### 15.7 Wildlife Monitoring

AI-enabled camera traps with edge inference. YOLOv5 on embedded devices with PIR motion sensors. Transformer-augmented YOLO achieves 94% mAP. Edge-device memory limits constrain large-scale deployment.

---

## 16. Open Problems and Future Directions

### 16.1 Adaptive Slicing (Active Research)
Current SAHI uses uniform grids. Research is converging on content-aware allocation:
- Density-guided (GOIS: +278% AP_small)
- Heatmap-guided (dual-resolution with attention maps)
- Altitude-aware (UAV-specific)
- RL-based placement (AdaZoom -- early stages)

**Gap:** No published work on using the detector's own internal attention maps to guide slicing placement.

### 16.2 End-to-End Learned Tiling
Current tiling is a separate pre-processing step. A fully end-to-end approach would jointly optimize tiling parameters and detection. This remains an open problem.

### 16.3 Temporal Tiling for Video
SAHI operates per-frame. For video, temporal coherence could reduce redundant computation -- objects don't teleport between frames. Combining SAHI with tracking (ByteTrack, BoT-SORT) is common practice but temporal slice optimization is unexplored.

### 16.4 NMS-Free Tiling
With NMS-free detectors (YOLO26, RT-DETR, DINO), the post-processing pipeline must change. These detectors produce single-box-per-object predictions, but tiled inference still generates duplicates from overlapping tiles. How to merge tile outputs from NMS-free detectors is an emerging question.

### 16.5 Sub-Pixel Objects
Datasets like EV-UAV (6.8x5.4px average) push beyond what tiling alone can address. Future work likely combines SAHI-style tiling with super-resolution, event cameras, or multi-frame aggregation.

---

## 17. References

### Foundational (Pre-2015)
- Rosenfeld & Thurston. "Edge and Curve Detection for Visual Scene Analysis." IEEE Trans. Computers, 1971.
- Viola & Jones. "Rapid Object Detection Using a Boosted Cascade of Simple Features." CVPR, 2001.
- Dalal & Triggs. "Histograms of Oriented Gradients for Human Detection." CVPR, 2005.
- Felzenszwalb et al. "Object Detection with Discriminatively Trained Part-Based Models." TPAMI, 2010.
- Uijlings et al. "Selective Search for Object Recognition." IJCV, 2013.

### Deep Learning Era (2014--2017)
- Girshick et al. "Rich Feature Hierarchies for Accurate Object Detection." CVPR, 2014.
- Girshick. "Fast R-CNN." ICCV, 2015.
- Ren et al. "Faster R-CNN: Towards Real-Time Object Detection with RPNs." NeurIPS, 2015.
- Liu et al. "SSD: Single Shot MultiBox Detector." ECCV, 2016.
- Redmon et al. "You Only Look Once." CVPR, 2016.
- Redmon & Farhadi. "YOLO9000: Better, Faster, Stronger." CVPR, 2017.
- Lin et al. "Feature Pyramid Networks for Object Detection." CVPR, 2017.
- Dai et al. "Deformable Convolutional Networks." ICCV, 2017.

### Small Object Detection (2017--2021)
- Li et al. "Perceptual Generative Adversarial Networks for Small Object Detection." CVPR, 2017.
- Bodla et al. "Soft-NMS -- Improving Object Detection With One Line of Code." ICCV, 2017.
- Hosang et al. "Learning Non-Maximum Suppression." CVPR, 2017.
- Singh & Davis. "An Analysis of Scale Invariance in Object Detection -- SNIP." CVPR, 2018.
- Singh et al. "SNIPER: Efficient Multi-Scale Training." NeurIPS, 2018.
- Zhang et al. "SOD-MTGAN." ECCV, 2018.
- Van Etten. "Satellite Imagery Multiscale Rapid Detection with Windowed Networks." WACV, 2019.
- Li et al. "Scale-Aware Trident Networks for Object Detection." ICCV, 2019.
- Yang et al. "Clustered Object Detection in Aerial Images." ICCV, 2019.
- Liu et al. "Adaptive NMS." CVPR, 2019.
- Zhu et al. "Deformable ConvNets v2." CVPR, 2019.
- Zheng et al. "Distance-IoU Loss." AAAI, 2020.
- Wang et al. "SOLOv2 (Matrix NMS)." NeurIPS, 2020.
- Solovyev et al. "Weighted Boxes Fusion." Image & Vision Computing, 2021.
- Luo et al. "NMS-Loss." ICMR, 2021.
- Zheng et al. "Enhancing Geometric Factors in Model Learning (Cluster-NMS)." IEEE T-Cybernetics, 2021.
- Wang et al. "EdgeDuet." IEEE INFOCOM, 2021.

### SAHI and Tiling (2020--2022)
- **Akyon, Altinuc, Temizel. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." IEEE ICIP, 2022. arXiv:2202.06934.**
- Yang et al. "QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection." CVPR, 2022 (Oral).

### Post-SAHI and Modern (2022--2025)
- Zhang et al. "DINO: DETR with Improved DeNoising Anchor Boxes." ICLR, 2023.
- Wang et al. "InternImage: Exploring Large-Scale Vision Foundation Models." CVPR, 2023 (Highlight).
- "Adaptive SAHI for Remote Sensing." Remote Sensing, 2023.
- Nguyen et al. "Dynamic Tiling." arXiv:2309.11069, 2023.
- Liu et al. "Grounding DINO." ECCV, 2024.
- Zhao et al. "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)." CVPR, 2024.
- "GOIS: Guided Object Inference Slicing." Neurocomputing, 2025.
- "Scene Heatmap-Guided Adaptive Tiling." MDPI Symmetry, 2025.
- "Tiling-Based Semantic Gating for Dense Object Detection." arXiv:2509.10779, 2025.
- Tian et al. "YOLOv12: Attention-Centric Real-Time Object Detectors." NeurIPS, 2025.
- "RF-DETR." ICLR, 2026 (submitted 2025).
- Ultralytics. "YOLO26." September 2025.

### Datasets
- Everingham et al. "The PASCAL Visual Object Classes Challenge." IJCV, 2010.
- Lin et al. "Microsoft COCO: Common Objects in Context." ECCV, 2014.
- Zhu et al. "VisDrone-DET2018/2019/2020/2021." ECCV/ICCV Workshops, 2018--2021.
- Lam et al. "xView: Objects in Context in Overhead Imagery." arXiv:1802.07856, 2018.
- Xia et al. "DOTA: A Large-scale Dataset for Object Detection in Aerial Images." CVPR, 2018.
- Wang et al. "AI-TOD: Tiny Object Detection in Aerial Images." ICPR, 2020.
- Yu et al. "Scale Match for Tiny Person Detection (TinyPerson)." WACV, 2020.
- Cheng et al. "SODA: Towards Open-World Small Object Detection." ICCV, 2023.

### Evaluation
- Bolya et al. "TIDE: A General Toolbox for Identifying Object Detection Errors." ECCV, 2020.

### TensorRT, Edge Devices, and Quantization
- NVIDIA. "How TensorRT Works." TensorRT Documentation, 2024.
- NVIDIA. "TensorRT Dynamic Shapes." TensorRT Documentation, 2024.
- NVIDIA. "Achieving FP32 Accuracy for INT8 Inference Using QAT with TensorRT." NVIDIA Developer Blog, 2021.
- Ding et al. "Reg-PTQ: Regression-specialized PTQ for Fully Quantized Object Detector." CVPR, 2024.
- Wang et al. "EdgeDuet: Tiling Small Object Detection for Edge Assisted Autonomous Mobile Vision." IEEE INFOCOM, 2021.
- Wang et al. "EdgeDuet (extended)." IEEE/ACM Transactions on Networking, 2022.
- Oro et al. "Work-Efficient Parallel Non-Maximum Suppression for Embedded GPU Architectures." The Computer Journal, 2025.
- Fluendo. "NMS-Raster: GPU-Based Non-Maximum Suppression." Blog, 2025.
- NVIDIA. "Native SAHI Plugins for DeepStream 8.0." NVIDIA Developer Forums, 2025.
- NVIDIA. "Jetson Modules and Performance Benchmarks." NVIDIA Embedded, 2025.
- Ultralytics. "TensorRT Integration." Ultralytics Documentation, 2025.
- Ultralytics. "Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO." 2025.
- pyke. "ort: ONNX Runtime Bindings for Rust." GitHub, 2025.
- TEXSAR. "Automated Drone Image Analysis Tool." texsar.org, 2025.

### Industry Applications
- "CCOD-Dataset: Crop Canopy Organ-Level Detection with SAHI + YOLOv10n." Remote Sensing, 2025.
- "AI-Driven Tethered Drone for Maritime Surveillance." Drones, 2025.
- "Edge AI for Industrial Visual Inspection." Algorithms, 2025.
- "Two-Stage Wildlife Event Classification for Edge Devices." Sensors, 2025.
- "Enhancing UAV Aerial Image Analysis: Integrating Advanced SAHI Techniques." ResearchGate, 2024.

---

*Last updated: April 2026. This survey accompanies the sahi.rs project -- a high-performance Rust implementation of SAHI.*
