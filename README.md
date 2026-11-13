# stable-diffusion-2-xl-streamlit

A *super minimal* Streamlit app for playing around with Stable Diffusion 2.1 and XL 1.0.

Includes:
- Text to image (txt2img) (v2.1 and XL 1.0)
- Image to image (img2img) (v2.1 and XL 1.0)
- Inpainting (v2.0 and XL 1.0)
- Negative prompt input for all methods.

Uses the 2.1 768px model by default. Can be changed to the base (512px) by changing the hardcoded model_id.

## Requirements

- Stable Diffusion generally requires a GPU with at least 8GB of memory.
- Make sure you have the Nvidia drivers and all that.

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

**Note:** Stability AI has removed the SD 2.x repositories (`stabilityai/stable-diffusion-2-1` and `stabilityai/stable-diffusion-2-inpainting`) from Hugging Face. The 2.1 and 2.0 options therefore only work if you already have those models in your local Hugging Face cache (or point the model IDs at a mirror). The XL 1.0, XL refiner and SDXL Turbo models still download automatically.

Images are automatically saved in an `outputs/` folder, along with their prompt.

### Out of memory errors

Out of memory errors occur frequently in my setup (8GB GPU), even though I've tried to limit the usage of cache and clean up the CUDA. They are quickly fixed with a clear cache and rerun in Streamlit, conveniently accessible on the C and R keys.

## Why

Mostly created this because I couldn't find a simple UI in the days right after the release of Stable Diffusion 2 (and also because I wanted to refresh my memory of Streamlit).

## License

Code in this repository, written by me, can be freely used under the [CC0 license](/LICENSE). This covers all code except where otherwise noted.

Always remember to check the license of the downloaded models as well :-)
