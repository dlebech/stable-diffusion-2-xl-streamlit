from typing import Optional

import streamlit as st
from PIL import Image

from sd2.generate import (
    DEFAULT_STEPS_AND_GUIDANCE,
    FLUX_2_KLEIN_4B,
    MODEL_DEFAULTS,
    MODEL_HINTS,
    PIPELINE_NAMES,
    generate,
    model_versions,
)

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


def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    negative_prompt = st.text_area(
        "Negative prompt",
        value="",
        key=f"{prefix}-negative-prompt",
    )
    # The version is part of the slider keys so each model gets its own defaults.
    version = kwargs.get("version")
    default_steps, default_guidance = MODEL_DEFAULTS.get(
        version, DEFAULT_STEPS_AND_GUIDANCE
    )
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider(
            "Number of inference steps",
            min_value=1,
            max_value=200,
            value=default_steps,
            key=f"{prefix}-{version}-inference-steps",
        )
    with col2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.0,
            max_value=20.0,
            value=default_guidance,
            step=0.5,
            key=f"{prefix}-{version}-guidance-scale",
        )
    enable_attention_slicing = st.checkbox(
        "Enable attention slicing (enables higher resolutions but is slower)",
        key=f"{prefix}-attention-slicing",
    )
    enable_cpu_offload = st.checkbox(
        "Enable CPU offload (if you run out of memory, e.g. for XL models; always on for Z-Image and FLUX.2 on GPUs with less than 24GB)",
        key=f"{prefix}-cpu-offload",
        value=False,
    )

    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            image = generate(
                prompt,
                pipeline_name,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                enable_attention_slicing=enable_attention_slicing,
                enable_cpu_offload=enable_cpu_offload,
                **kwargs,
            )
            set_image(OUTPUT_IMAGE_KEY, image.copy())
        st.image(image)


def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=64,
            max_value=1600,
            step=16,
            value=768,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=64,
            max_value=1600,
            step=16,
            value=768,
            key=f"{prefix}-height",
        )
    return width, height


def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        return image

    return get_image(LOADED_IMAGE_KEY)


def inpainting():
    image = image_uploader("inpainting")

    if not image:
        return None, None

    brush_size = st.number_input("Brush Size", value=50, min_value=1, max_value=100)

    # Imported here because streamlit-drawable-canvas (>=0.13) is a Streamlit
    # v2 component that can only be registered inside a running Streamlit app.
    from streamlit_drawable_canvas import st_canvas

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
        return_image_data=True,
        key="inpainting-canvas",
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


def model_selectbox(prefix, pipeline_name: PIPELINE_NAMES):
    version = st.selectbox(
        "Model version", model_versions(pipeline_name), key=f"{prefix}-version"
    )
    if version in MODEL_HINTS:
        st.caption(MODEL_HINTS[version])
    if version == FLUX_2_KLEIN_4B and pipeline_name != "txt2img":
        st.caption(
            "FLUX.2 [klein] edits the input image, so describe the change in the prompt, e.g. \"replace the dog with a cat\"."
            + (" Strength is ignored." if pipeline_name == "img2img" else "")
        )
    return version


def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    version = model_selectbox(prefix, "txt2img")
    st.markdown(
        "**Note**: Z-Image and FLUX.2 need CPU offload on 16GB GPUs (turned on automatically). XL 1.0 output can be refined afterwards with img2img"
    )
    prompt_and_generate_button(
        prefix, "txt2img", width=width, height=height, version=version
    )


def inpainting_tab():
    prefix = "inpaint"
    col1, col2 = st.columns(2)

    with col1:
        image_input, mask_input = inpainting()

    with col2:
        if image_input and mask_input:
            version = model_selectbox(prefix, "inpaint")
            strength = st.slider(
                "Strength of inpainting (1.0 essentially ignores the masked area of the original input image)",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix,
                "inpaint",
                image_input=image_input,
                mask_input=mask_input,
                version=version,
                strength=strength,
            )


def img2img_tab():
    prefix = "img2img"
    col1, col2 = st.columns(2)

    with col1:
        image = image_uploader(prefix)
        if image:
            st.image(image)

    with col2:
        if image:
            version = model_selectbox(prefix, "img2img")
            strength = st.slider(
                "Strength (1.0 ignores the existing image so it's not a useful value)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix, "img2img", image_input=image, version=version, strength=strength
            )


def main():
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion, Z-Image & FLUX.2 Simple Playground")

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
            if st.button("Use this image for img2img"):
                set_image(LOADED_IMAGE_KEY, output_image.copy())
                st.rerun()
            st.markdown(
                "The button should also work for inpainting. However, there is a bug in the inpainting canvas so clicking the button will sometimes work for inpainting and sometimes not. It depends on whether you have previously uploaded an image in inpainting."
            )
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
