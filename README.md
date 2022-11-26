# stable-diffusion-2-streamlit

A *super minimal* Streamlit app for playing around with Stable Diffusion 2.

Includes:
- Text to image (txt2img)
- Image to image (img2img)
- Inpainting

## Install & run

### Requirements

- Stable Diffusion generally requires a GPU with at least 8GB of memory.
- Make sure you have the Nvidia drivers and all that

Personally, I'm running this on Ubuntu 22.04 with an RTX 2070 GPU (8GB memory) using Python 3.10 in a fresh conda environment

### Install

```
pip install -r requirements.txt
streamlit run sd2/main.py
```

The first time it runs, it will download the model from Hugging Face automatically.

Images are automatical.ly saved in and `outputs/` folder, along with their prompt.

## Why

Mostly created this because I couldn't find a simple UI in the days right after the release of Stable Diffusion 2 (and also because I wanted to refresh my memory of Streamlit).