import random
import torch
from pathlib import Path
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLImg2ImgPipeline
from rembg import remove, new_session


# =====================================================
# 1. 配置
# =====================================================

BASE_MODEL = "./models/RealVisXL_V5.0_fp16.safetensors"

LORA_DIR = "./models/tomato/stem"

REFERENCE_DIR = Path("./seed/tomato/stem")
OUTPUT_DIR = Path("./output/tomato_stem_results")

IMAGES_PER_REFERENCE = 10

WIDTH = 1024
HEIGHT = 1536

LORA_WEIGHT = 0.8
STRENGTH = 1.0

STEPS = 35
CFG = 5.5

BASE_SEED = random.randint(0, 2**31 - 1)

SAVE_RAW = True
REMOVE_BACKGROUND = True


# =====================================================
# 2. 不同植株结构的 Prompt
# =====================================================

SHAPE_PROMPTS = [
    "tall plant, spaced branches",
    "dense green foliage",
    "asymmetric branching",
    "long lateral branches",
    "compact branches, broad leaves",
    "varied leaf sizes",
    "slender plant, sparse foliage",
    "moderately dense compound leaves",
]

PROMPT_SUFFIX = (
    "tmtstem, single whole tomato plant, "
    "upright green stem, natural side branches, "
    "green compound serrated leaves, "
    "visible fibrous roots, "
    "no fruits, no flowers, "
    "full plant visible, pure white background, "
    "photorealistic botanical image"
)

NEGATIVE_PROMPT = (
    "multiple plants, multiple stems growing separately, "
    "tomato fruits, red tomatoes, flowers, "
    "flowerpot, soil, greenhouse, garden, "
    "cut off roots, cropped plant, "
    "deformed branches, fused leaves, "
    "blurry, text, watermark"
)


# =====================================================
# 3. 输出目录
# =====================================================

WHITE_DIR = OUTPUT_DIR / "white"

WHITE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 4. 读取参考图片
# =====================================================

references = sorted(
    p for p in REFERENCE_DIR.iterdir()
    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
)

if not references:
    raise RuntimeError("没有找到seed图片")



# =====================================================
# 5. 加载基础模型
# =====================================================

print("\nLoading RealVisXL V5.0...")

pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    BASE_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.to("cuda")


# =====================================================
# 6. 加载 LoRA
# =====================================================

candidates = [
    "tomato_stem_lora.safetensors",
]

lora_file = next(
    (name for name in candidates if (LORA_DIR / name).exists()),
    None
)

if lora_file is None:
    raise FileNotFoundError(
        f"找不到LoRA，请检查目录: {LORA_DIR}"
    )

print("Loading LoRA:", lora_file)

pipe.load_lora_weights(
    str(LORA_DIR),
    weight_name=lora_file,
    adapter_name="stem"
)

pipe.set_adapters(
    ["stem"],
    adapter_weights=[LORA_WEIGHT]
)

pipe.set_progress_bar_config(disable=False)


# =====================================================
# 7. 抠图模型
# =====================================================

bg_session = None

if REMOVE_BACKGROUND:
    print("\nLoading background removal model...")
    bg_session = new_session("isnet-general-use")


# =====================================================
# 8. 参考图预处理
# =====================================================

def prepare_reference(path):

    image = Image.open(path).convert("RGB")

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
# 9. 自动抠图并合成纯白背景
# =====================================================

def make_white_background(image):

    cutout = remove(
        image.convert("RGB"),
        session=bg_session,
        alpha_matting=False,
        post_process_mask=True
    ).convert("RGBA")

    white_bg = Image.new(
        "RGBA",
        image.size,
        (255, 255, 255, 255)
    )

    white_bg.alpha_composite(cutout)

    return white_bg.convert("RGB")


# =====================================================
# 10. 批量生成
# =====================================================

total = len(references) * IMAGES_PER_REFERENCE
count = 0

print("\n====================================")
print("Tomato Stem Generation")
print("====================================")


for ref_index, reference_path in enumerate(references):

    reference_image = prepare_reference(reference_path)

    for variant in range(IMAGES_PER_REFERENCE):

        count += 1
        seed = BASE_SEED + count - 1

        shape = SHAPE_PROMPTS[
            (count - 1) % len(SHAPE_PROMPTS)
        ]

        prompt = f"{shape}, {PROMPT_SUFFIX}"


        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        with torch.inference_mode():

            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=reference_image,
                strength=STRENGTH,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=generator
            ).images[0]

        name = (
            f"ref{ref_index+1:02d}_"
            f"var{variant+1:02d}_"
            f"seed{seed}"
        )


        # 白底处理
        if REMOVE_BACKGROUND:
            image = make_white_background(image)

        image.save(WHITE_DIR / f"{name}.png")

        print("Saved:", WHITE_DIR / f"{name}.png")


print("\n====================================")
print("Generation completed!")
print("White:", WHITE_DIR)
print("====================================")

