# InfiniteFarm-Code

InfiniteFarm-Code is a modular framework for generating biologically plausible 3D crop models using generative AI and growth-informed spatial estimation.

The framework consists of three independently executable modules:

- **Module 1: 2D Crop Image Generation** — Generates high-quality crop images with white backgrounds using diffusion models and crop-specific LoRA weights.
- **Module 2: Image-to-3D Reconstruction** — Converts generated 2D crop images into textured 3D models using TRELLIS.2.
- **Module 3: Growth-informed Spatial Estimation and Biological Assembly** — Assembles independently reconstructed plant and fruit models into complete 3D crops using morphological measurements and statistical sampling.

Each module can be executed independently using the corresponding input files and pretrained models.

## 1. Environment Setup

### 1.1 Hardware Requirements

The framework has been developed and tested on the following hardware configuration:

| Component | Configuration |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| GPU Memory | 40 GB |
| System Memory | 72 GB |
| Operating System | Linux |
| GPU Acceleration | CUDA |

A CUDA-compatible NVIDIA GPU is required for the generative models. GPU memory requirements may vary depending on the module and model configuration.

### 1.2 Installation

Clone the repository:

```bash
git clone https://github.com/langyiningbit/InfiniteFarm-Code.git
cd InfiniteFarm-Code
```

The three modules have different software dependencies and can be configured independently.

**Module 1 — 2D Image Generation**

The primary dependencies include:

- Python
- PyTorch
- Diffusers
- Transformers
- Accelerate
- Pillow
- rembg

**Module 2 — Image-to-3D Reconstruction**

Module 2 is based on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

Please follow the official TRELLIS.2 installation instructions to configure the reconstruction environment and install the required dependencies.

**Module 3 — Spatial Estimation and Assembly**

The primary dependencies include:

- Python
- NumPy
- SciPy
- Trimesh
- Open3D
- PyOpenGL
- EGL-compatible rendering environment

The exact dependency versions should be configured according to the corresponding module scripts and upstream model requirements.

## 2. Module 1: 2D Crop Image Generation

Module 1 generates crop-specific 2D images using diffusion models with LoRA adaptation.

Taking **tomato** as an example, two scripts are provided to independently generate tomato plant images and tomato fruit images.

### 2.1 Tomato Plant Generation

Run:

```bash
cd module1
python generate_stem.py
```

The script generates tomato plant images with white backgrounds, including stems, branches, and leaves.

### 2.2 Tomato Fruit Generation

Run:

```bash
python generate_tomato.py
```

The script generates tomato fruit images with white backgrounds.

Both scripts can be configured to generate multiple images with variations in appearance.

**Output:** 2D crop images that can be used as inputs to Module 2.

## 3. Module 2: Image-to-3D Reconstruction

Module 2 reconstructs textured 3D crop components from 2D images using TRELLIS.2.

Taking tomato as an example, the generated tomato plant and tomato fruit images are reconstructed independently.

Run:

```bash
cd module2
python example.py
```

The reconstruction produces two types of 3D assets:

- **Tomato plant model (`stem.glb`)** — A 3D model containing the vegetative structures of the tomato plant.
- **Tomato fruit model (`tomato.glb`)** — A 3D model representing an individual tomato fruit.

These models can be used independently or passed to Module 3 for biological assembly.

**Output:** Textured 3D crop components in GLB format.

## 4. Module 3: Growth-informed Spatial Estimation and Biological Assembly

Module 3 assembles independently reconstructed 3D crop components into complete plant models with biologically plausible dimensions, fruit quantities, and spatial arrangements.

The module uses crop-specific morphological measurements to guide statistical sampling, component scaling, and fruit placement.

### 4.1 Input Files

For the tomato example, the required inputs are:

| File | Description |
|---|---|
| `stem.glb` | Reconstructed 3D tomato plant model |
| `tomato.glb` | Reconstructed 3D tomato fruit model |
| `measurements.csv` | Crop-specific morphological measurements and spatial observations |

### 4.2 Execution

Run the following command from the directory containing `run_module3.py`:

```bash
EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl \
python run_module3.py \
    --stem stem.glb \
    --fruit tomato.glb \
    --measurements measurements.csv \
    --crop tomato \
    --stage mature \
    --output-dir output \
    --resolution 768 \
    --frames 120 \
    --fps 15
```

### 4.3 Parameters

| Parameter | Description |
|---|---|
| `--stem` | Path to the reconstructed plant GLB model |
| `--fruit` | Path to the reconstructed fruit GLB model |
| `--measurements` | Path to the morphological measurement CSV file |
| `--crop` | Crop category |
| `--stage` | Growth stage |
| `--output-dir` | Directory for generated outputs |
| `--resolution` | Rendering resolution |
| `--frames` | Number of video frames |
| `--fps` | Video frame rate |

### 4.4 Output

Module 3 generates:

1. **Complete 3D plant model (GLB):** A biologically assembled crop model containing the plant structure and attached fruits.
2. **Visualization video (MP4):** A rendered video showing the generated 3D crop model.

The assembly process uses morphological statistics to control plant dimensions, fruit sizes, fruit quantities, and attachment positions.

## 5. Pretrained Models and Example Data

The pretrained models required by the three modules, together with the example input files for Module 3, are available through Google Drive.

**[Download Pretrained Models and Example Data](https://drive.google.com/drive/folders/1OsMsKh3m-_2hv_2msAqAOrzKokBtZsR6?usp=drive_link)**

The shared resources include the model files used by the generation and reconstruction modules, as well as the example GLB assets and morphological measurement data required to execute Module 3.

Users can run each module independently using the corresponding pretrained models and example data, without executing the entire pipeline.

## 6. Repository

Source code is available at:

https://github.com/langyiningbit/InfiniteFarm-Code

The framework is organized into three independent modules to facilitate reproducibility, individual component evaluation, and future extensions to additional crop categories.
