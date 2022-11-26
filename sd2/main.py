import datetime
import os
import re
from typing import Literal, Optional

import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image

PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img"]
DEFAULT_PROMPT = "border collie puppy"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


@st.cache(allow_output_mutation=True, max_entries=1)
def get_pipeline(name: PIPELINE_NAMES):
    if name in ["txt2img", "img2img"]:
        model_id = "stabilityai/stable-diffusion-2-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        if name == "img2img":
            pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
        pipe = pipe.to("cuda")
        return pipe
    elif name == "inpaint":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        return pipe


def generate(prompt, pipeline_name: PIPELINE_NAMES, image_input=None, mask_input=None):
    """Generates an image based on the given prompt and pipeline name"""
    if pipeline_name == "inpaint" and image_input and mask_input:
        pipe = get_pipeline(pipeline_name)
        image = pipe(prompt=prompt, image=image_input, mask_image=mask_input).images[0]
    elif pipeline_name == "txt2img":
        pipe = get_pipeline(pipeline_name)
        image = pipe(prompt).images[0]
    elif pipeline_name == "img2img" and image_input:
        pipe = get_pipeline(pipeline_name)
        image = pipe(prompt, init_image=image_input).images[0]
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    os.makedirs("outputs", exist_ok=True)

    filename = (
        "outputs/"
        + re.sub(r"\s+", "_", prompt)[:50]
        + f"_{datetime.datetime.now().timestamp()}"
    )
    image.save(f"{filename}.png")
    set_image(OUTPUT_IMAGE_KEY, image.copy())
    with open(f"{filename}.txt", "w") as f:
        f.write(prompt)
    return image


def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            image = generate(prompt, pipeline_name, **kwargs)
        st.image(image)


def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        image = image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        return image

    return get_image(LOADED_IMAGE_KEY)


def inpainting():
    image = image_uploader("inpainting")

    if not image:
        return None, None

    brush_size = st.number_input("Brush Size", value=50, min_value=1, max_value=100)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=brush_size,
        stroke_color="#FFFFFF",
        background_color="#000000",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        # Use repr(image) to force the component to reload when the image
        # changes, i.e. when asking to use the current output image
        key="inpainting",
    )

    if not canvas_result or canvas_result.image_data is None:
        return None, None

    mask = canvas_result.image_data
    mask = mask[:, :, -1] > 0
    if mask.sum() > 0:
        mask = Image.fromarray(mask)
        st.image(mask)
        return image, mask

    return None, None


def txt2img_tab():
    prompt_and_generate_button("txt2img", "txt2img")


def inpainting_tab():
    col1, col2 = st.columns(2)

    with col1:
        image_input, mask_input = inpainting()

    with col2:
        if image_input and mask_input:
            prompt_and_generate_button(
                "inpaint", "inpaint", image_input=image_input, mask_input=mask_input
            )


def img2img_tab():
    col1, col2 = st.columns(2)

    with col1:
        image = image_uploader("img2img")
        if image:
            st.image(image)

    with col2:
        if image:
            prompt_and_generate_button("img2img", "img2img", image_input=image)


def main():
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion 2.0 Simple Playground")
    tab1, tab2, tab3 = st.tabs(
        ["Text to Image (txt2img)", "Inpainting", "Image to image (img2img)"]
    )

    with tab1:
        txt2img_tab()

    with tab2:
        inpainting_tab()

    with tab3:
        img2img_tab()

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_image(OUTPUT_IMAGE_KEY)
        if output_image:
            st.image(output_image)
            if st.button("Use this image for inpainting and img2img"):
                set_image(LOADED_IMAGE_KEY, output_image.copy())
                st.experimental_rerun()
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
