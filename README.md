# stable-diffusion-2-streamlit

A *super minimal* Streamlit app for playing around with Stable Diffusion 2.

Includes:
- Text to image (txt2img) (v2.1)
- Image to image (img2img) (v2.1)
- Inpainting (v2.0)
- Negative prompt input for all methods.

Uses the 768px model if possible. Can be changed to the base (512px) by changing the hardcoded model_id.

## Requirements

- Stable Diffusion generally requires a GPU with at least 8GB of memory.
- Make sure you have the Nvidia drivers and all that.

Observations based on my own machine:
- Using Ubuntu 22.04 with an RTX 2070 GPU (8GB memory)
- Using Python 3.10 in a fresh conda environment.
- The 768 model works but it is only possible with xformers *or* attention slicing.
- With xformers enabled, 768x768 pixel images are possible

## Install & Run

```
pip install -r requirements.txt
streamlit run main.py
```

The first time it runs, it will download the model from Hugging Face automatically.

Images are automatical.ly saved in and `outputs/` folder, along with their prompt.

### Out of memory errors

Out of memory errors occur frequently in my setup (8GB GPU), even though I've tried to limit the usage of cache and clean up the CUDA. They are quickly fixed with a clear cache and rerun in Streamlit, conveniently accessible on the C and R keys.

## Why

Mostly created this because I couldn't find a simple UI in the days right after the release of Stable Diffusion 2 (and also because I wanted to refresh my memory of Streamlit).

## License

Code written by me can be freely used under the [CC0 license](/LICENSE). This covers all code except where otherwise noted.

Always remember to check the license of the downloaded models as well :-)
