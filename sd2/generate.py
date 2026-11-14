import gc
import datetime
import inspect
import os
import re
from typing import Literal

import streamlit as st
import torch
from diffusers import (
    DiffusionPipeline,
    Flux2KleinInpaintPipeline,
    Flux2KleinPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ZImageImg2ImgPipeline,
    ZImageInpaintPipeline,
    ZImagePipeline,
)

PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img"]
Z_IMAGE_TURBO = "Z-Image Turbo"
FLUX_2_KLEIN_4B = "FLUX.2 [klein] 4B"
SD_XL_10 = "XL 1.0"
SD_XL_10_REFINER = "XL 1.0 refiner"
SD_XL_TURBO = "SDXL Turbo"
MODEL_VERSIONS = Literal[
    "Z-Image Turbo",
    "FLUX.2 [klein] 4B",
    "XL 1.0",
    "XL 1.0 refiner",
    "SDXL Turbo",
]

SDXL_VERSIONS = (SD_XL_10, SD_XL_10_REFINER, SD_XL_TURBO)

_SDXL_KWARGS = dict(dtype=torch.float16, use_safetensors=True, variant="fp16")
_BF16 = dict(dtype=torch.bfloat16)

# (pipeline name, model version) -> (pipeline class, model id, from_pretrained kwargs).
# The order here is the order of the model options in each tab of the UI.
PIPELINES = {
    ("txt2img", Z_IMAGE_TURBO): (ZImagePipeline, "Tongyi-MAI/Z-Image-Turbo", _BF16),
    ("txt2img", FLUX_2_KLEIN_4B): (Flux2KleinPipeline, "black-forest-labs/FLUX.2-klein-4B", _BF16),
    ("txt2img", SD_XL_10): (StableDiffusionXLPipeline, "stabilityai/stable-diffusion-xl-base-1.0", _SDXL_KWARGS),
    ("txt2img", SD_XL_TURBO): (StableDiffusionXLPipeline, "stabilityai/sdxl-turbo", _SDXL_KWARGS),
    ("img2img", Z_IMAGE_TURBO): (ZImageImg2ImgPipeline, "Tongyi-MAI/Z-Image-Turbo", _BF16),
    # FLUX.2 [klein] has no strength-based img2img; the regular pipeline uses the image as an editing reference.
    ("img2img", FLUX_2_KLEIN_4B): (Flux2KleinPipeline, "black-forest-labs/FLUX.2-klein-4B", _BF16),
    ("img2img", SD_XL_10_REFINER): (StableDiffusionXLImg2ImgPipeline, "stabilityai/stable-diffusion-xl-refiner-1.0", _SDXL_KWARGS),
    ("inpaint", Z_IMAGE_TURBO): (ZImageInpaintPipeline, "Tongyi-MAI/Z-Image-Turbo", _BF16),
    ("inpaint", FLUX_2_KLEIN_4B): (Flux2KleinInpaintPipeline, "black-forest-labs/FLUX.2-klein-4B", _BF16),
    ("inpaint", SD_XL_10): (StableDiffusionXLInpaintPipeline, "stabilityai/stable-diffusion-xl-base-1.0", _SDXL_KWARGS),
}

# Short usage hints shown below the model selector.
MODEL_HINTS = {
    Z_IMAGE_TURBO: "Turbo model: 8-9 steps, guidance 0 (at guidance 0 the negative prompt has no effect). CPU offload is turned on automatically on GPUs with less than 24GB.",
    FLUX_2_KLEIN_4B: "Distilled model: 4 steps; guidance and negative prompt are ignored. CPU offload is turned on automatically on GPUs with less than 24GB.",
    SD_XL_TURBO: "Turbo model: 1-4 steps, guidance 0.",
}

# Default (steps, guidance scale) per model; everything else uses DEFAULT_STEPS_AND_GUIDANCE.
DEFAULT_STEPS_AND_GUIDANCE = (20, 7.5)
MODEL_DEFAULTS = {
    Z_IMAGE_TURBO: (9, 0.0),
    FLUX_2_KLEIN_4B: (4, 1.0),
    SD_XL_TURBO: (2, 0.0),
}

# Z-Image and FLUX.2 don't fit in GPU memory without CPU offload below this.
OFFLOAD_REQUIRED_BELOW_BYTES = 24 * 2**30


def model_versions(name: PIPELINE_NAMES) -> list[str]:
    """The model versions available for a pipeline, default first."""
    return [version for (n, version) in PIPELINES if n == name]


@st.cache_resource(max_entries=1)
def get_pipeline(
    name: PIPELINE_NAMES,
    version: MODEL_VERSIONS = Z_IMAGE_TURBO,
    enable_cpu_offload=False,
) -> DiffusionPipeline:
    if (name, version) not in PIPELINES:
        raise Exception(f"Could not find pipeline {name} and version {version}")

    pipeline_class, model_id, kwargs = PIPELINES[(name, version)]
    pipe = pipeline_class.from_pretrained(model_id, **kwargs)
    # Potential speedup for XL, but didn't work super great for me.
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    if enable_cpu_offload:
        print("Enabling CPU offload for pipeline")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    return pipe


def generate(
    prompt,
    pipeline_name: PIPELINE_NAMES,
    image_input=None,
    mask_input=None,
    negative_prompt=None,
    steps=50,
    width=768,
    height=768,
    guidance_scale=7.5,
    enable_attention_slicing=False,
    enable_cpu_offload=False,
    version=Z_IMAGE_TURBO,
    strength=1.0,
):
    """Generates an image based on the given prompt and pipeline name"""
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)

    def callback(pipe, step, timestep, callback_kwargs):
        p.progress(step / steps)
        return callback_kwargs

    # NOTE: This code is not being used, since the combined XL 1.0 and refiner
    # pipeline did not work on my hardware.
    refiner_version = None
    try:
        version, refiner_version = version.split(" + ")
    except:
        pass

    if (
        version not in SDXL_VERSIONS
        and not enable_cpu_offload
        and torch.cuda.get_device_properties(0).total_memory < OFFLOAD_REQUIRED_BELOW_BYTES
    ):
        print(f"Enabling CPU offload for {version} (needs more than 24GB of GPU memory otherwise)")
        enable_cpu_offload = True

    pipe = get_pipeline(
        pipeline_name, version=version, enable_cpu_offload=enable_cpu_offload
    )
    torch.cuda.empty_cache()

    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        callback_on_step_end=callback,
        guidance_scale=guidance_scale,
    )

    if pipeline_name == "inpaint" and image_input and mask_input:
        kwargs.update(image=image_input, mask_image=mask_input, strength=strength)
    elif pipeline_name == "txt2img":
        kwargs.update(width=width, height=height)
    elif pipeline_name == "img2img" and image_input:
        kwargs.update(image=image_input, strength=strength)
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    if pipeline_name != "txt2img" and version not in SDXL_VERSIONS:
        # The newer pipelines default to 1024x1024 output and need sizes
        # divisible by 16, so keep the input image's size (and aspect ratio).
        kwargs.update(
            width=image_input.width // 16 * 16, height=image_input.height // 16 * 16
        )

    # Not every pipeline supports every option (e.g. FLUX.2 has no negative prompt).
    supported = inspect.signature(pipe.__call__).parameters
    for key in [k for k in kwargs if k not in supported]:
        if kwargs[key] is not None:
            print(f"{type(pipe).__name__} does not support {key}, ignoring it")
        del kwargs[key]

    print("kwargs", kwargs)

    high_noise_frac = 0.8

    # When a refiner is being used, we need to add the denoising_end/start parameters.
    # NOTE: This code is not being used, since the combined XL 1.0 and refiner
    # pipeline did not work on my hardware.
    if refiner_version:
        images = pipe(
            denoising_end=high_noise_frac, output_type="latent", **kwargs
        ).images
    else:
        images = pipe(**kwargs).images

    if refiner_version:
        images = images.detach().clone()
        pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        refiner = get_pipeline(
            pipeline_name,
            version=refiner_version,
            enable_cpu_offload=enable_cpu_offload,
        )
        kwargs.pop("width", None)
        kwargs.pop("height", None)
        images = refiner(image=images, denoising_start=high_noise_frac, **kwargs).images
    image = images[0]

    os.makedirs("outputs", exist_ok=True)

    filename = (
        "outputs/"
        + re.sub(r"\s+", "_", prompt)[:50]
        + f"_{datetime.datetime.now().timestamp()}"
    )
    image.save(f"{filename}.png")
    with open(f"{filename}.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\nNegative Prompt: {negative_prompt}")
    return image
