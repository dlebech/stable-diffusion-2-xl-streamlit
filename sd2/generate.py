import datetime
import os
import random
import re
from typing import Literal, Union

import streamlit as st
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from stable_diffusion_videos import StableDiffusionWalkPipeline

PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img", "video"]
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO_DIR = "outputs/video"


@st.cache(allow_output_mutation=True, max_entries=1)
def get_pipeline(
    name: PIPELINE_NAMES,
) -> Union[
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionWalkPipeline,
]:
    if name in ["txt2img", "img2img"]:
        model_id = "stabilityai/stable-diffusion-2-1-base"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        # pipe = StableDiffusionPipeline.from_pretrained(
        #    model_id, torch_dtype=torch.float16
        # )
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if name == "img2img":
            pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
        pipe = pipe.to("cuda")
        return pipe
    elif name == "inpaint":
        model_id = "stabilityai/stable-diffusion-2-inpainting"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        return pipe
    elif name == "video":
        model_id = "stabilityai/stable-diffusion-2-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        pipe = StableDiffusionWalkPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            feature_extractor=None,
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        return pipe


def generate(
    prompt,
    pipeline_name: PIPELINE_NAMES,
    image_input=None,
    mask_input=None,
    negative_prompt=None,
    prompt_to=None,
    width=512,
    height=512,
):
    """Generates an image based on the given prompt and pipeline name"""
    steps = 50
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)
    callback = lambda step, *_: p.progress(step / steps)

    pipe = get_pipeline(pipeline_name)
    torch.cuda.empty_cache()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        callback=callback,
    )

    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    if pipeline_name == "inpaint" and image_input and mask_input:
        kwargs.update(image=image_input, mask_image=mask_input)
    elif pipeline_name == "txt2img":
        kwargs.update(width=width, height=height)
    elif pipeline_name == "img2img" and image_input:
        kwargs.update(
            init_image=image_input,
        )
    elif pipeline_name == "video" and prompt_to:
        video_path = pipe.walk(
            prompts=[prompt, prompt_to],
            seeds=[random.randint(1, 10_000), random.randint(1, 10_000)],
            num_inference_steps=20,
            num_interpolation_steps=20,
            width=width,
            height=height,
            output_dir=OUTPUT_VIDEO_DIR,
            name="test2",
        )
        return video_path
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    with torch.autocast("cuda"):
        image = pipe(**kwargs).images[0]

    filename = (
        OUTPUT_DIR
        + "/"
        + re.sub(r"\s+", "_", prompt)[:50]
        + f"_{datetime.datetime.now().timestamp()}"
    )
    image.save(f"{filename}.png")
    with open(f"{filename}.txt", "w") as f:
        f.write(prompt)
    return image
