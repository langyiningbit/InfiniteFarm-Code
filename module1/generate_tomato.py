import os
import random
import torch

from pathlib import Path
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLImg2ImgPipeline
from rembg import remove, new_session


# =====================================================
# 1. 基础配置
# =====================================================

BASE_MODEL = "./models/RealVisXL_V5.0_fp16.safetensors"

LORA_DIR = "./models/tomato/fruit"
LORA_FILE = "tomato_fruit_lora.safetensors"


REFERENCE_DIR = "./seed/tomato/fruit"
OUTPUT_DIR = "./output/tomato_fruit_results"


IMAGES_PER_REFERENCE = 10

WIDTH = 1024
HEIGHT = 1024

LORA_WEIGHT = 0.8


STRENGTH = 1.0

STEPS = 35
CFG = 5.5

BASE_SEED = random.randint(0, 2**31 - 1)

SAVE_RAW = True
REMOVE_BACKGROUND = True


# =====================================================
# 2. Prompt
# =====================================================

PROMPT = (
    "tmtfruit, "
    "a realistic photograph of a single whole ripe red tomato, "
    "natural round tomato shape, "
    "smooth red skin with subtle natural texture, "
    "small green calyx on top, "
    "one isolated tomato centered in the image, "
    "plain white background, "
    "high quality product photography"
)

NEGATIVE_PROMPT = (
    "multiple tomatoes, pile of tomatoes, "
    "tomato slices, cut tomato, half tomato, "
    "tomato plant, vines, branches, "
    "deformed fruit, fused fruits, malformed shape, "
    "cropped, close-up, basket, table, "
    "text, watermark, illustration"
)


# =====================================================
# 3. 创建输出目录
# =====================================================

output = Path(OUTPUT_DIR)

final_dir = output / "white"


final_dir.mkdir(parents=True, exist_ok=True)


# =====================================================
# 4. 读取seed图片
# =====================================================

reference_dir = Path(REFERENCE_DIR)

references = sorted(
    p for p in reference_dir.iterdir()
    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
)

if not references:
    raise RuntimeError("没有找到seed图片，请检查 REFERENCE_DIR")




# =====================================================
# 5. 加载 RealVisXL
# =====================================================

print("\nLoading RealVisXL V5.0...")

pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    BASE_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.to("cuda")

print("Loading Tomato LoRA...")

pipe.load_lora_weights(
    LORA_DIR,
    weight_name=LORA_FILE,
    adapter_name="tomato"
)

pipe.set_adapters(
    ["tomato"],
    adapter_weights=[LORA_WEIGHT]
)

pipe.set_progress_bar_config(disable=False)


# =====================================================
# 6. 加载抠图模型
# =====================================================

bg_session = None

if REMOVE_BACKGROUND:
    print("\nLoading background removal model...")
    bg_session = new_session("isnet-general-use")


# =====================================================
# 7. 图片预处理
# =====================================================

def prepare_reference(path):

    image = Image.open(path).convert("RGB")

    # 保持比例，放在白色正方形画布中
    image = ImageOps.contain(
        image,
        (WIDTH, HEIGHT),
        method=Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (255, 255, 255)
    )

    x = (WIDTH - image.width) // 2
    y = (HEIGHT - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


# =====================================================
# 8. 自动抠图 + 白底
# =====================================================

def make_white_background(image):

    cutout = remove(
        image.convert("RGB"),
        session=bg_session,
        alpha_matting=False,
        post_process_mask=True
    ).convert("RGBA")

    white = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    white.alpha_composite(cutout)

    return white.convert("RGB")


# =====================================================
# 9. 开始批量生成
# =====================================================

total = len(references) * IMAGES_PER_REFERENCE

print("\n====================================")
print("Starting Generation")
print("====================================")

count = 0

for ref_index, reference_path in enumerate(references):


    reference_image = prepare_reference(reference_path)

    for variant in range(IMAGES_PER_REFERENCE):

        count += 1

        seed = BASE_SEED + count - 1

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        with torch.inference_mode():

            image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=reference_image,
                strength=STRENGTH,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=generator
            ).images[0]

        name = (
            f"ref{ref_index + 1:02d}_"
            f"var{variant + 1:02d}_"
            f"seed{seed}"
        )


        # 自动抠图
        if REMOVE_BACKGROUND:
            image = make_white_background(image)

        # 保存最终图片
        image.save(final_dir / f"{name}.png")

        print("Saved:", final_dir / f"{name}.png")

print("\n====================================")
print("All done!")
print("images:", final_dir)
print("====================================")
