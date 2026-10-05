# Project Explanation --- CNN vs Fine-Tuned SRGAN for Single-Image Super-Resolution

## 1. Project Overview

### Project title

**Comparative Study of CNN-Based and GAN-Based Single-Image
Super-Resolution**

### Short description

This project builds and compares two deep-learning approaches for
restoring a high-resolution (HR) image from a low-resolution (LR) image:

1.  A conventional **CNN-based super-resolution model** used as the
    baseline.
2.  A **pretrained SRGAN (Super-Resolution Generative Adversarial
    Network)** that is fine-tuned on the project's dataset.

The main goal is not simply to make an image larger. The goal is to
learn how to **reconstruct missing high-frequency visual information**
such as edges, textures and fine details.

The project investigates the difference between:

-   pixel/reconstruction-oriented super-resolution, and
-   perceptual/adversarial super-resolution.

The final system should produce a controlled experimental comparison
using the **same LR-HR data, same scale factor, and the same test set**.

------------------------------------------------------------------------

## 2. Core Research Question

The central research question is:

> **How does fine-tuning a pretrained GAN-based super-resolution model
> affect image reconstruction and perceptual quality compared with a
> conventional CNN-based super-resolution model?**

The project should investigate both numerical and visual performance.

Important sub-questions:

-   Does the CNN produce higher PSNR/SSIM?
-   Does SRGAN produce sharper and more realistic textures?
-   Does fine-tuning improve a pretrained SRGAN on the target dataset?
-   Is there a trade-off between pixel-level accuracy and perceptual
    quality?
-   How much additional computation does SRGAN require compared with the
    CNN?
-   How does inference time differ between the two approaches?

Do not assume in advance that SRGAN will have higher PSNR or SSIM. The
experiments must determine this.

------------------------------------------------------------------------

# 3. What Is Image Super-Resolution?

Single-image super-resolution (SISR) attempts to reconstruct a
high-resolution image from a single low-resolution image.

Example:

``` text
Input:
64 × 64 LR image

        ↓

Super-resolution model

        ↓

Output:
256 × 256 SR image
```

For this project, the primary scale factor should be **4×**.

The dataset will contain high-resolution images. Low-resolution versions
will be generated from them using a controlled degradation/downsampling
process.

Example:

``` text
Original HR
256 × 256
     |
     | 4× downsampling
     v
LR
64 × 64
```

The original 256×256 image becomes the ground-truth target.

Therefore every training example is:

``` text
LR image  →  Ground-truth HR image
64×64     →  256×256
```

------------------------------------------------------------------------

# 4. Why HR Images Are Collected Instead of Only LR Images

The project uses supervised learning.

If only an LR image is available, there is no known answer telling the
model what the correct HR image should be.

Instead, collect or use HR images and generate corresponding LR images.

``` text
HR image
   |
   +---- degradation/downsampling ----> LR image
   |
   +----------------------------------> Ground-truth HR
```

This produces paired training examples:

``` text
Input = LR
Target = original HR
```

This is also useful because both models can be evaluated fairly using
exactly the same data.

------------------------------------------------------------------------

# 5. Project Architecture

The project has two main models.

## 5.1 CNN Baseline

The CNN model receives an LR image and predicts an HR image.

``` text
LR Image
   |
   v
Feature Extraction
   |
   v
Nonlinear Mapping
   |
   v
Upsampling
   |
   v
Reconstruction
   |
   v
SR Image
```

A compact **SRCNN-style or lightweight CNN super-resolution model**
should be used as the baseline.

The CNN should be trained from scratch.

The purpose of this model is not to compete with the latest
super-resolution architectures. It is a controlled conventional
deep-learning baseline against which the GAN-based approach can be
compared.

------------------------------------------------------------------------

# 6. SRGAN Model

SRGAN contains two networks:

## 6.1 Generator

The Generator performs super-resolution.

``` text
LR image
   |
   v
Generator
   |
   v
Generated SR image
```

The Generator should be based on the pretrained SRGAN-compatible
architecture used by the selected checkpoint.

The project should NOT arbitrarily mix incompatible pretrained weights
and architectures.

Before loading a checkpoint, verify:

-   architecture
-   number of channels
-   scale factor
-   residual block structure
-   upsampling method
-   expected input/output size
-   state-dict compatibility

------------------------------------------------------------------------

## 6.2 Discriminator

The Discriminator receives either:

-   a real HR image, or
-   a generated SR image.

It attempts to determine whether the image is real or generated.

``` text
                    ┌──────────────┐
LR ────────────────>│  Generator   │
                    └──────┬───────┘
                           |
                           v
                       SR image
                           |
                           v
                    ┌──────────────┐
Real HR ────────────>│ Discriminator│
SR image ───────────>│              │
                    └──────────────┘
```

The Discriminator encourages the Generator to produce images that look
more realistic.

------------------------------------------------------------------------

# 7. Why Fine-Tune Instead of Training SRGAN Completely From Scratch?

The project should use a **pretrained SRGAN-compatible checkpoint**
rather than attempting to learn the entire GAN from random
initialization.

Reasons:

1.  GAN training is significantly more difficult to stabilize.
2.  SRGAN requires a useful Generator before adversarial training
    becomes effective.
3.  The available laptop has an RTX 4050 Laptop GPU with 6 GB VRAM.
4.  Fine-tuning reduces training time and compute requirements.
5.  Pretrained weights already contain useful low-level and mid-level
    image representations.
6.  The project can focus more on experimentation and analysis instead
    of spending most of the project on expensive GAN pretraining.

This does NOT mean simply running a pretrained model.

The SRGAN must be **fine-tuned on the project's training data**.

The project should document:

``` text
Pretrained SRGAN
       |
       v
Target dataset
       |
       v
Fine-tuning
       |
       v
Domain-adapted SRGAN
```

------------------------------------------------------------------------

# 8. Important Research Methodology

The project should contain at least these experimental stages.

## Experiment A --- Bicubic Baseline

Use traditional bicubic interpolation.

``` text
LR → Bicubic → HR-sized image
```

This is the simplest baseline.

It establishes how much improvement the learned models provide.

------------------------------------------------------------------------

## Experiment B --- CNN/SRCNN

Train the CNN from scratch.

``` text
LR → CNN → SR
```

Use reconstruction loss such as L1 or MSE.

------------------------------------------------------------------------

## Experiment C --- Pretrained SRGAN

Run the pretrained SRGAN on the test set before fine-tuning.

This establishes the performance of the original pretrained model on the
target data.

------------------------------------------------------------------------

## Experiment D --- Fine-Tuned SRGAN

Fine-tune the pretrained SRGAN on the training dataset.

``` text
Pretrained SRGAN
       |
       v
Fine-tuning
       |
       v
Fine-tuned SRGAN
```

Evaluate on the same held-out test set.

------------------------------------------------------------------------

## Final comparison

At minimum compare:

``` text
Bicubic
   vs
CNN/SRCNN
   vs
Pretrained SRGAN
   vs
Fine-Tuned SRGAN
```

This gives the project a genuine experimental component.

------------------------------------------------------------------------

# 9. Dataset Strategy

## Recommended starting size

Target approximately:

**2,000--5,000 high-resolution images**

The project does not need millions of unique images.

Training should use random crops/patches so each HR image can produce
many training samples.

Example:

``` text
One 512×512 HR image
        |
        +--> 256×256 crop
        +--> 256×256 crop
        +--> 256×256 crop
        +--> ...
```

Each HR patch is degraded to produce its LR counterpart.

Example for 4× SR:

``` text
HR patch = 256×256
LR patch = 64×64
```

------------------------------------------------------------------------

# 10. Dataset Split

Use a strict image-level split:

``` text
70% Training
15% Validation
15% Testing
```

Example for 4,000 unique HR images:

``` text
2,800 train
600 validation
600 test
```

Important:

**Split images BEFORE generating patches.**

Do not generate thousands of patches first and then randomly split
patches. That can cause data leakage because patches from the same
original image could appear in both training and testing.

Correct:

``` text
Original images
       |
       +---- Train images → training patches
       |
       +---- Validation images → validation patches
       |
       +---- Test images → test patches
```

------------------------------------------------------------------------

# 11. Degradation Pipeline

For the first version, keep degradation controlled and reproducible.

Primary pipeline:

``` text
HR image
   |
   +--> crop
   |
   +--> resize/downsample ×4
   |
   v
LR image
```

Use a clearly documented interpolation method such as bicubic for the
initial experiment.

Optional later extension:

``` text
HR
 |
 +--> blur
 +--> resize
 +--> noise
 +--> JPEG compression
 |
 v
LR
```

Do not add complicated degradation initially. Establish a clean baseline
first.

------------------------------------------------------------------------

# 12. Image Sizes

Recommended initial setup:

### Training

``` text
LR patch: 64 × 64
HR patch: 256 × 256
Scale: 4×
```

### Evaluation

Images can be evaluated at a fixed HR resolution or using appropriately
sized full-resolution images.

Training should use patches because full-resolution GAN training would
consume substantially more VRAM.

------------------------------------------------------------------------

# 13. Data Augmentation

Recommended lightweight augmentations:

-   horizontal flip
-   vertical flip if appropriate for the dataset
-   90° rotation if appropriate
-   random crop

Avoid transformations that change the actual super-resolution task
unless intentionally studying them.

All augmentations must be applied consistently to the HR image before
generating its LR counterpart.

Example:

``` text
HR
 |
 +--> random crop
 +--> horizontal flip
 |
 v
HR patch
 |
 v
downsample
 |
 v
LR patch
```

------------------------------------------------------------------------

# 14. Loss Functions

## CNN baseline

Start with L1 loss:

``` text
L_CNN = mean(|SR - HR|)
```

L1 is often preferable to pure MSE for image reconstruction because it
can produce less over-smoothed results.

MSE can also be implemented as an optional experiment.

------------------------------------------------------------------------

## SRGAN

The SRGAN Generator should use a combination of:

### Content/perceptual loss

Measures whether the generated image preserves meaningful visual
information relative to the HR target.

### Adversarial loss

Encourages generated images to look realistic to the Discriminator.

Conceptually:

``` text
Generator loss
=
content/perceptual component
+
adversarial component
```

The exact formulation must match the selected SRGAN
implementation/paper/checkpoint.

Do not invent a loss formulation while claiming it is the original SRGAN
loss.

------------------------------------------------------------------------

# 15. Perceptual Loss

Perceptual loss compares feature representations instead of only
comparing raw pixels.

A pretrained feature extractor such as VGG can be used.

Conceptually:

``` text
HR ──────────> VGG ──> feature representation
                         |
                         | compare
                         |
SR ──────────> VGG ──> feature representation
```

This helps the Generator preserve perceptually meaningful structures.

If the selected pretrained SRGAN implementation already provides its
perceptual-loss implementation, reuse the compatible implementation
rather than creating a different loss without documenting the change.

------------------------------------------------------------------------

# 16. Evaluation Metrics

Use multiple metrics because no single metric captures super-resolution
quality completely.

## PSNR

Measures pixel-level reconstruction quality.

Higher is generally better.

However, high PSNR does not automatically mean an image looks more
realistic.

------------------------------------------------------------------------

## SSIM

Measures structural similarity.

Higher is generally better.

------------------------------------------------------------------------

## LPIPS

Measures perceptual similarity using deep feature representations.

Lower is generally better.

LPIPS is particularly useful for this project because GAN-based
super-resolution may produce visually convincing textures without
maximizing PSNR.

------------------------------------------------------------------------

## Inference time

Measure how long each model takes to process the same test image.

Report:

-   average inference time
-   images per second if useful
-   model parameter count

------------------------------------------------------------------------

# 17. Visual Evaluation

Numerical metrics are not sufficient.

For a fixed set of test images, create comparison grids:

``` text
Ground Truth | Bicubic | CNN | Pretrained SRGAN | Fine-Tuned SRGAN
```

Use the same images and same crop regions.

Include zoomed-in regions showing:

-   edges
-   textures
-   text
-   hair
-   buildings
-   repetitive patterns
-   other high-frequency details

Do not cherry-pick only successful examples.

Include representative examples and clearly document the selection
method.

------------------------------------------------------------------------

# 18. Expected Trade-Off

Do not assume that the SRGAN will always achieve the best PSNR.

A conventional CNN optimized for reconstruction can produce smoother
outputs that are closer to the ground truth at a pixel level.

A GAN may generate sharper and more perceptually realistic textures.

Therefore, the project should explicitly investigate the possible
trade-off:

``` text
Pixel fidelity
      ↕
Perceptual realism
```

A result where CNN has higher PSNR but SRGAN looks sharper is not
automatically a failure. It is an important finding to analyze.

------------------------------------------------------------------------

# 19. Recommended Technology Stack

## Programming language

**Python 3.11 or a compatible version supported by the chosen
PyTorch/CUDA setup.**

------------------------------------------------------------------------

## Deep learning

**PyTorch**

Primary framework for:

-   model implementation
-   training
-   GPU acceleration
-   checkpointing
-   mixed precision

------------------------------------------------------------------------

## Computer vision

Recommended:

-   Pillow
-   OpenCV

Use Pillow for straightforward image loading/transformation and OpenCV
where useful for image processing and visualization.

------------------------------------------------------------------------

## Numerical processing

-   NumPy

------------------------------------------------------------------------

## Metrics

Possible stack:

-   `torchmetrics`
-   `scikit-image`
-   `lpips`

Use established implementations rather than manually implementing
complex metrics unless the project specifically requires it.

------------------------------------------------------------------------

## Visualization

-   Matplotlib

Optional:

-   Seaborn for non-image statistical plots

------------------------------------------------------------------------

## Experiment tracking

Recommended:

-   TensorBoard

Optionally:

-   Weights & Biases

Do not make external experiment-tracking services mandatory. The project
must work locally.

------------------------------------------------------------------------

## Configuration

Use YAML configuration files so hyperparameters are not hard-coded
throughout Python files.

Example:

``` yaml
scale_factor: 4
hr_patch_size: 256
lr_patch_size: 64
batch_size: 2
num_workers: 2
learning_rate: 0.0001
epochs: 50
mixed_precision: true
```

Actual values should be tuned based on VRAM and experiments.

------------------------------------------------------------------------

# 20. Hardware Constraints

Development machine:

``` text
CPU: AMD Ryzen 7 7445HS
RAM: 16 GB
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB
Storage: approximately 1 TB
```

The training pipeline must be designed around **6 GB VRAM**.

Recommended:

-   64×64 LR / 256×256 HR training patches
-   batch size starting at 2
-   mixed precision / AMP
-   efficient DataLoader settings
-   gradient accumulation only if needed
-   avoid loading the entire dataset into RAM
-   checkpoint regularly
-   release unnecessary GPU tensors
-   use `torch.cuda.amp` / modern PyTorch AMP APIs as appropriate for
    the installed version

The code must automatically detect CUDA and fall back to CPU if CUDA is
unavailable, although CPU training is not the intended training
environment.

------------------------------------------------------------------------

# 21. Training Strategy

## CNN

Train from scratch.

Suggested initial workflow:

``` text
1. Load HR images
2. Split train/val/test
3. Generate HR patches
4. Generate LR patches
5. Train CNN
6. Validate after every epoch
7. Save best checkpoint based on validation metric
8. Evaluate on test set
```

------------------------------------------------------------------------

## SRGAN

Recommended workflow:

``` text
1. Load compatible pretrained SRGAN checkpoint
2. Verify architecture and checkpoint compatibility
3. Evaluate pretrained checkpoint
4. Fine-tune on training dataset
5. Validate regularly
6. Save best fine-tuned checkpoint
7. Evaluate on test set
```

If the selected implementation requires separate Generator/Discriminator
checkpoints, support them explicitly.

------------------------------------------------------------------------

# 22. Checkpointing

Every training run should save:

-   model weights
-   optimizer state
-   scheduler state if used
-   epoch
-   validation metrics
-   configuration/hyperparameters

Example:

``` text
checkpoints/
├── cnn/
│   ├── best.pt
│   └── last.pt
└── srgan/
    ├── best_generator.pt
    ├── best_discriminator.pt
    └── last.pt
```

Do not overwrite the best checkpoint with every epoch.

------------------------------------------------------------------------

# 23. Reproducibility

Set random seeds for:

-   Python
-   NumPy
-   PyTorch

Document:

-   dataset version
-   train/validation/test split
-   random seed
-   scale factor
-   patch size
-   batch size
-   learning rate
-   optimizer
-   scheduler
-   number of epochs
-   pretrained checkpoint used
-   GPU
-   PyTorch version
-   CUDA version

The same experiment should be reproducible as closely as practical.

------------------------------------------------------------------------

# 24. Recommended Project Structure

Use a modular structure rather than putting everything into one file.

``` text
super_resolution_project/
│
├── README.md
├── project_explanation.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── cnn.yaml
│   └── srgan.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── cnn_sr.py
│   ├── srgan_generator.py
│   └── srgan_discriminator.py
│
├── datasets/
│   ├── sr_dataset.py
│   └── degradation.py
│
├── losses/
│   ├── reconstruction.py
│   ├── perceptual.py
│   └── adversarial.py
│
├── training/
│   ├── train_cnn.py
│   ├── train_srgan.py
│   └── trainer_utils.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── visual_comparison.py
│
├── inference/
│   └── super_resolve.py
│
├── utils/
│   ├── config.py
│   ├── seed.py
│   ├── checkpoint.py
│   └── image_utils.py
│
├── checkpoints/
│
├── results/
│   ├── metrics/
│   ├── comparisons/
│   └── plots/
│
└── notebooks/
    └── exploration.ipynb
```

The exact structure can be simplified if the project becomes
unnecessarily fragmented.

------------------------------------------------------------------------

# 25. Important Implementation Rules

## Rule 1 --- Keep CNN and SRGAN data identical

Both models must receive equivalent LR-HR pairs.

Do not use one dataset for CNN and another for SRGAN unless that
difference is itself an experiment.

------------------------------------------------------------------------

## Rule 2 --- Keep test data untouched

Never fine-tune on the test set.

Never use test images for hyperparameter selection.

------------------------------------------------------------------------

## Rule 3 --- Avoid data leakage

Split at the original-image level before generating patches.

------------------------------------------------------------------------

## Rule 4 --- Track every experiment

Each run should have:

``` text
experiment name
model
dataset
seed
hyperparameters
checkpoint
metrics
```

------------------------------------------------------------------------

## Rule 5 --- Do not silently change pretrained architecture

If a pretrained checkpoint is used, its architecture must match.

If the project changes the architecture, document that as a deliberate
modification and do not pretend it is the original SRGAN.

------------------------------------------------------------------------

# 26. Research Papers to Study

The literature review should begin with these papers.

## Paper 1 --- SRCNN

**Learning a Deep Convolutional Network for Image Super-Resolution**

Dong et al.

Purpose:

-   CNN-based super-resolution foundation
-   establishes the conventional CNN baseline

------------------------------------------------------------------------

## Paper 2 --- SRGAN

**Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network**

Ledig et al., CVPR 2017.

Purpose:

-   GAN-based super-resolution
-   Generator/Discriminator formulation
-   perceptual/content loss
-   adversarial loss
-   perceptual realism

------------------------------------------------------------------------

## Paper 3 --- ESRGAN

**ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks**

Wang et al.

Purpose:

-   understand improvements over SRGAN
-   RRDB
-   improved perceptual quality
-   relativistic discriminator
-   removal of BatchNorm in the RRDB architecture

ESRGAN should initially be treated as an optional extension, not a
mandatory third model.

------------------------------------------------------------------------

## Optional Paper 4 --- Real-ESRGAN

**Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
Synthetic Data**

Wang et al.

Purpose:

-   understand realistic degradation
-   blind real-world super-resolution
-   potential future extension

Do not implement Real-ESRGAN in the first version of the project.

------------------------------------------------------------------------

# 27. What Counts as the Project's Contribution?

The project contribution is not claiming that SRGAN itself is new.

The contribution should be framed as an **experimental comparative study
and adaptation**.

Potential contribution:

1.  Implement a CNN-based baseline.
2.  Adapt a pretrained SRGAN to the selected dataset.
3.  Compare pretrained vs fine-tuned SRGAN.
4.  Compare both against a conventional CNN.
5.  Evaluate pixel-level and perceptual metrics.
6.  Analyze the quality/realism trade-off.
7.  Analyze computational cost.

An optional extension can introduce a small architectural or degradation
modification and evaluate it through an ablation study.

Do not claim a novel research architecture unless the project actually
introduces and experimentally validates one.

------------------------------------------------------------------------

# 28. Suggested Final Experiments

Minimum viable experiment set:

### Experiment 1

``` text
Bicubic
```

### Experiment 2

``` text
CNN/SRCNN trained from scratch
```

### Experiment 3

``` text
Pretrained SRGAN without fine-tuning
```

### Experiment 4

``` text
Fine-tuned SRGAN
```

### Experiment 5 --- Optional

``` text
Fine-tuned SRGAN + one documented modification
```

The fifth experiment should only be added after the first four are
stable.

------------------------------------------------------------------------

# 29. Final Results Table

The project should eventually produce a table similar to:

  -------------------------------------------------------------------------------
  Model            PSNR ↑     SSIM ↑    LPIPS ↓   Parameters        Avg. Notes
                                                               Inference 
                                                                    Time 
  ------------ ---------- ---------- ---------- ------------ ----------- --------
  Bicubic                                                N/A             

  CNN/SRCNN                                                              

  Pretrained                                                             
  SRGAN                                                                  

  Fine-Tuned                                                             
  SRGAN                                                                  

  Optional                                                               
  Modified                                                               
  SRGAN                                                                  
  -------------------------------------------------------------------------------

Do not fill this table with expected or fabricated values. Populate it
only after running experiments.

------------------------------------------------------------------------

# 30. Expected Output Artifacts

The coding agent should ultimately produce:

1.  Working dataset preparation pipeline.
2.  CNN training pipeline.
3.  SRGAN fine-tuning pipeline.
4.  Evaluation pipeline.
5.  Inference script.
6.  Model checkpoints.
7.  Metric CSV/JSON files.
8.  Visual comparison images.
9.  Training curves.
10. Final comparison tables.
11. Reproducible configuration files.
12. Documentation explaining the methodology.

------------------------------------------------------------------------

# 31. Inference Workflow

The final user-facing inference process should be:

``` text
Input LR image
      |
      v
Select model
      |
      +---- CNN
      |
      +---- Fine-Tuned SRGAN
      |
      v
Generate SR image
      |
      v
Save/display result
```

Example:

``` bash
python inference/super_resolve.py \
    --model srgan \
    --checkpoint checkpoints/srgan/best_generator.pt \
    --input input.jpg \
    --output output_sr.png
```

The exact CLI can be designed by the coding agent.

------------------------------------------------------------------------

# 32. Optional Demo

After the core training/evaluation pipeline is stable, an optional
lightweight UI can be created.

Possible interface:

``` text
Upload LR image
      |
      +---- Select CNN
      |
      +---- Select Fine-Tuned SRGAN
      |
      v
Compare outputs
```

The UI is secondary. Do not spend significant project time on frontend
development before the ML experiments are complete.

------------------------------------------------------------------------

# 33. Development Order

The coding agent should implement the project in this order:

## Phase 1 --- Environment

-   Set up Python environment.
-   Install PyTorch with the correct CUDA build.
-   Verify `torch.cuda.is_available()`.
-   Print GPU name and VRAM.
-   Verify AMP support.

## Phase 2 --- Dataset

-   Select dataset.
-   Download/prepare HR images.
-   Split into train/validation/test.
-   Implement paired LR generation.
-   Implement patch extraction.
-   Visualize LR/HR pairs.

## Phase 3 --- CNN

-   Implement CNN/SRCNN.
-   Implement dataset loader.
-   Implement training loop.
-   Add validation.
-   Add checkpointing.
-   Train baseline.
-   Evaluate.

## Phase 4 --- SRGAN

-   Select a compatible pretrained SRGAN implementation/checkpoint.
-   Verify architecture.
-   Load checkpoint.
-   Run inference before fine-tuning.
-   Verify output quality.

## Phase 5 --- Fine-tuning

-   Implement SRGAN fine-tuning.
-   Use AMP.
-   Add checkpointing.
-   Validate during training.
-   Save best Generator/Discriminator.

## Phase 6 --- Evaluation

Run the same test set through:

``` text
Bicubic
CNN
Pretrained SRGAN
Fine-Tuned SRGAN
```

Calculate:

-   PSNR
-   SSIM
-   LPIPS
-   inference time
-   parameter count

## Phase 7 --- Visualization

Generate consistent comparison grids.

## Phase 8 --- Analysis

Create:

-   metric comparison
-   training curves
-   qualitative comparison
-   computational comparison
-   failure-case analysis

## Phase 9 --- Optional extension

Only after the baseline experiments are complete, add one modification
and perform an ablation study.

------------------------------------------------------------------------

# 34. What the Coding Agent Must NOT Do

Do not:

-   train on the test set
-   randomly split patches after extraction
-   fabricate metric values
-   claim pretrained inference is fine-tuning
-   claim a pretrained model is implemented from scratch
-   mix incompatible checkpoints and architectures
-   use full-resolution images as GPU training tensors unnecessarily
-   hard-code Windows-specific paths
-   assume a specific GPU is always available
-   silently resize ground truth after model prediction
-   compare models using different test images
-   optimize only for PSNR and ignore perceptual quality
-   introduce ESRGAN/Real-ESRGAN complexity before the basic experiment
    works
-   download arbitrary checkpoints without documenting their source and
    architecture
-   make the UI before the ML pipeline is functional

------------------------------------------------------------------------

# 35. Definition of Done

The project is considered complete when:

-   [ ] Dataset preparation works reproducibly.
-   [ ] LR-HR pairs are generated correctly.
-   [ ] Train/validation/test split has no image leakage.
-   [ ] CNN baseline trains successfully.
-   [ ] CNN checkpoint can be loaded for inference.
-   [ ] Pretrained SRGAN checkpoint is verified.
-   [ ] Pretrained SRGAN inference works.
-   [ ] SRGAN can be fine-tuned on the project's dataset.
-   [ ] Fine-tuned checkpoint can be loaded independently.
-   [ ] Bicubic/CNN/pretrained-SRGAN/fine-tuned-SRGAN are evaluated on
    exactly the same test set.
-   [ ] PSNR is calculated.
-   [ ] SSIM is calculated.
-   [ ] LPIPS is calculated.
-   [ ] Inference time is measured consistently.
-   [ ] Visual comparison grids are generated.
-   [ ] Training curves are saved.
-   [ ] Experiment configuration is reproducible.
-   [ ] No test-data leakage exists.
-   [ ] Results are not fabricated.
-   [ ] Final analysis discusses both numerical and perceptual quality.
-   [ ] The README explains how to reproduce training and inference.

------------------------------------------------------------------------

# 36. Final Project Concept in One Diagram

``` text
                         HR DATASET
                              |
                              v
                    Image-level split
                              |
             ┌────────────────┼────────────────┐
             |                |                |
           TRAIN             VAL              TEST
             |                                 |
             v                                 |
       Generate LR                            |
             |                                 |
             v                                 |
      LR-HR training pairs                    |
             |                                 |
       ┌─────┴──────────┐                      |
       |                |                      |
       v                v                      |
   CNN/SRCNN        Pretrained SRGAN           |
   from scratch          |                     |
       |                  v                     |
       |              Fine-tuning              |
       |                  |                     |
       v                  v                     |
   CNN SR output    Fine-Tuned SRGAN output    |
       |                  |                     |
       └──────────┬───────┘                     |
                  |                             |
                  v                             v
             SAME TEST SET <───────────────────┘
                  |
        ┌─────────┼──────────┐
        v         v          v
      PSNR      SSIM       LPIPS
        |         |          |
        └─────────┼──────────┘
                  v
       Visual + Computational
             Comparison
                  |
                  v
          Research Findings
```

## Overall Recommendation

The intended first version of this project is:

> **Train a conventional CNN/SRCNN super-resolution baseline from
> scratch, then fine-tune a compatible pretrained SRGAN on the same
> dataset and compare Bicubic, CNN, pretrained SRGAN, and fine-tuned
> SRGAN using PSNR, SSIM, LPIPS, inference time, and qualitative visual
> analysis.**

The project should prioritize **correct experimental methodology,
reproducibility, and analysis** over adding unnecessary model
complexity.
