import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import platform
import fooocus_version
import argparse

from modules.launch_util import is_installed, run, python, \
    run_pip, repo_dir, git_clone, requirements_met, script_path, dir_repos
from modules.model_loader import load_file_from_url
from modules.path import modelfile_path, lorafile_path, clip_vision_path, controlnet_path, vae_approx_path, fooocus_expansion_path, upscale_models_path


REINSTALL_ALL = False
DEFAULT_ARGS = ['--disable-smart-memory', '--disable-cuda-malloc']

def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.21')

    comfy_repo = os.environ.get('COMFY_REPO', "https://github.com/comfyanonymous/ComfyUI")
    comfy_commit_hash = os.environ.get('COMFY_COMMIT_HASH', "2381d36e6db8e8150e42ff2ede628db5b00ae26f")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    comfyui_name = 'ComfyUI-from-StabilityAI-Official'
    git_clone(comfy_repo, repo_dir(comfyui_name), "Inference Engine", comfy_commit_hash)
    sys.path.append(os.path.join(script_path, dir_repos, comfyui_name))

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if REINSTALL_ALL or not is_installed("xformers"):
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
            else:
                print("Installation of xformers is not supported in this version of Python.")
                print(
                    "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


model_filenames = [
    ('realisticStockPhoto_v10.safetensors',
     'https://civitai.com/api/download/models/154593')
]

lora_filenames = [
    ('epiCRealismHelper.safetensors',
     'https://civitai.com/api/download/models/118945?type=Model&format=SafeTensor')
]

clip_vision_filenames = [
    ('clip_vision_g.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors')
]

controlnet_filenames = [
    ('control-lora-canny-rank128.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors'),
    ('control-lora-canny-rank256.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors'),
    ('control-lora-depth-rank128.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors'),
    ('control-lora-depth-rank256.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors')
]

vae_approx_filenames = [
    ('xlvaeapp.pth',
     'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('taesd_decoder.pth',
     'https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth')
]


upscaler_filenames = [
    ('fooocus_upscaler_s409985e5.bin',
     'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin')
]


def download_models():
    for file_name, url in model_filenames:
        load_file_from_url(url=url, model_dir=modelfile_path, file_name=file_name)
    for file_name, url in lora_filenames:
        load_file_from_url(url=url, model_dir=lorafile_path, file_name=file_name)
    for file_name, url in clip_vision_filenames:
        load_file_from_url(url=url, model_dir=clip_vision_path, file_name=file_name)
    for file_name, url in controlnet_filenames:
        load_file_from_url(url=url, model_dir=controlnet_path, file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)
    for file_name, url in upscaler_filenames:
        load_file_from_url(url=url, model_dir=upscale_models_path, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=fooocus_expansion_path,
        file_name='pytorch_model.bin'
    )

    return


def parse_args():
    argv = sys.argv + DEFAULT_ARGS
    sys.argv = [sys.argv[0]]
    import comfy.cli_args
    sys.argv = argv

    parser = argparse.ArgumentParser('launch.py', parents=[comfy.cli_args.parser], conflict_handler='resolve')
    parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
    parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
    parser.add_argument("--listen", type=str, default=None, metavar="IP", nargs="?", const="0.0.0.0", help="Set the listen interface.")

    comfy.cli_args.args = parser.parse_args()


def cuda_malloc():
    import cuda_malloc


prepare_environment()

parse_args()

download_models()

from webui import *
