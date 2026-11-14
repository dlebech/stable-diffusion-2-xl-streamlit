# stable-diffusion-2-xl-streamlit

A *super minimal* Streamlit app for playing around with open image generation models: Z-Image Turbo, FLUX.2 [klein] and Stable Diffusion XL 1.0. (It started out as a Stable Diffusion 2 app, hence the name.)

Includes:
- Text to image (txt2img): Z-Image Turbo, FLUX.2 [klein] 4B, XL 1.0, SDXL Turbo
- Image to image (img2img): Z-Image Turbo, FLUX.2 [klein] 4B, XL 1.0 refiner
- Inpainting: Z-Image Turbo, FLUX.2 [klein] 4B, XL 1.0
- Negative prompt input for all methods (ignored by the models that don't support it, e.g. FLUX.2).

Uses Z-Image Turbo by default. The steps and guidance sliders start at sensible values for each model (the turbo/distilled models want few steps, around 2-9, and guidance 0; FLUX.2 ignores guidance), the txt2img width and height start at the model's native size (1024x1024, 512x512 for SDXL Turbo), and the app shows a short hint for each.

FLUX.2 [klein] works like an image editor in img2img and inpainting: it uses the input image as a reference, so the prompt should describe the change ("replace the dog with a cat"), and the img2img strength is ignored.

**Note:** Stable Diffusion 2.0/2.1 were dropped because Stability AI removed those repositories from Hugging Face. Stable Diffusion 3.5 is left out because its repositories are gated behind a licence agreement.

### Models and licences

| Model | Hugging Face repo | Licence |
|---|---|---|
| Z-Image Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Apache 2.0 |
| FLUX.2 [klein] 4B | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0 |
| XL 1.0, XL 1.0 refiner | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) | CreativeML OpenRAIL++-M |
| SDXL Turbo | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) | Stability AI Non-Commercial Research Community License (`sai-nc-community`) |

## Requirements

- Z-Image Turbo needs a 16GB GPU (with CPU offload, ~12GB peak), FLUX.2 [klein] 4B ~8GB with CPU offload, and the XL models run on 8GB.
- Make sure you have the Nvidia drivers and all that.
- The newer models are bigger. On a 16GB GPU (RTX 4070 Ti SUPER), Z-Image Turbo and FLUX.2 [klein] both need CPU offload, which the app turns on automatically for them on GPUs with less than 24GB; it brings peak GPU memory down to roughly 12GB for Z-Image Turbo and 8GB for FLUX.2 [klein] 4B. You also need plenty of system RAM for the offloaded parts (32GB+). The XL models run without offload on 16GB.
- Disk: Z-Image Turbo is roughly 33GB and FLUX.2 [klein] 4B roughly 16GB.

Observations based on my own machine:
- Using Ubuntu 22.04 with an RTX 2070 GPU (8GB memory)
- Using Python 3.10 in a fresh conda environment.
- The 2.1 768 model works out of the box with 768x768 images.
- For the XL 1.0 model, I personally have to use CPU offloading. Model CPU offloading is enough so it doesn't slow down inference so much.
  - With CPU offloading, I have been able to generate 1024x1024 images using the XL model.

## Install & Run

Requires Python 3.12 (pinned in `.python-version` and the version it's tested with; `requires-python` allows newer, which may work). Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`):

```
uv sync
uv run streamlit run main.py
```

`uv sync` creates a `.venv` with Python 3.12 (uv downloads it if needed).

Alternatively with pip, using the `requirements.txt` exported from the lock file:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

### GPU / CUDA wheels

On Linux and Windows, PyTorch is installed from the official PyTorch wheel index with CUDA 13.2 builds (`https://download.pytorch.org/whl/cu132`). This is configured in `pyproject.toml` for uv, and `requirements.txt` includes it as an `--extra-index-url` for pip. The CUDA libraries come bundled with the wheels, you only need a recent Nvidia driver (CUDA 13 needs driver 580 or newer, check with `nvidia-smi`). If you need a different CUDA version, change the index URL in `pyproject.toml` and run `uv lock` (not every PyTorch release is built for every CUDA version, so you may also have to relax the `torch`/`torchvision` versions).

To regenerate `requirements.txt` after changing dependencies:

```
uv export --no-hashes --no-dev --emit-index-url --format requirements-txt -o requirements.txt
```

The first time it runs, it will download the model from Hugging Face automatically.

Images are automatically saved in an `outputs/` folder, each with a `.txt` file next to it that records the settings used: prompt, negative prompt, pipeline (txt2img/img2img/inpaint), model, steps, guidance scale, output size, strength (img2img/inpaint) and whether CPU offload was on.

### Out of memory errors

Out of memory errors occur frequently in my setup (8GB GPU), even though I've tried to limit the usage of cache and clean up the CUDA. They are quickly fixed with a clear cache and rerun in Streamlit, conveniently accessible on the C and R keys.

## Why

Mostly created this because I couldn't find a simple UI in the days right after the release of Stable Diffusion 2 (and also because I wanted to refresh my memory of Streamlit).

## License

Code in this repository, written by me, can be freely used under the [CC0 license](/LICENSE). This covers all code except where otherwise noted.

Always remember to check the license of the downloaded models as well :-)
