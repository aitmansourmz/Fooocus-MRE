import os
import json

from modules.model_loader import load_file_from_url


def load_paths(paths_filename):
    paths_dict = {
        'modelfile_path': '../models/checkpoints/',
        'lorafile_path': '../models/loras/',
        'embeddings_path': '../models/embeddings/',
        'clip_vision_path': '../models/clip_vision/',
        'controlnet_path': '../models/controlnet/',
        'vae_approx_path': '../models/vae_approx/',
        'fooocus_expansion_path': '../models/prompt_expansion/fooocus_expansion/',
        'upscale_models_path': '../models/upscale_models/',
        'inpaint_models_path': '../models/inpaint/',
        'styles_path': '../sdxl_styles/',
        'wildcards_path': '../wildcards/',
        'temp_outputs_path': '../outputs/'
    }

    if os.path.exists(paths_filename):
        with open(paths_filename, encoding='utf-8') as paths_file:
            try:
                paths_obj = json.load(paths_file)
                if 'path_checkpoints' in paths_obj:
                    paths_dict['modelfile_path'] = paths_obj['path_checkpoints']
                if 'path_loras' in paths_obj:
                    paths_dict['lorafile_path'] = paths_obj['path_loras']
                if 'path_embeddings' in paths_obj:
                    paths_dict['embeddings_path'] = paths_obj['path_embeddings']
                if 'path_clip_vision' in paths_obj:
                    paths_dict['clip_vision_path'] = paths_obj['path_clip_vision']
                if 'path_controlnet' in paths_obj:
                    paths_dict['controlnet_path'] = paths_obj['path_controlnet']
                if 'path_vae_approx' in paths_obj:
                    paths_dict['vae_approx_path'] = paths_obj['path_vae_approx']
                if 'path_fooocus_expansion' in paths_obj:
                    paths_dict['fooocus_expansion_path'] = paths_obj['path_fooocus_expansion']
                if 'path_upscale_models' in paths_obj:
                    paths_dict['upscale_models_path'] = paths_obj['path_upscale_models']
                if 'path_inpaint_models' in paths_obj:
                    paths_dict['inpaint_models_path'] = paths_obj['path_inpaint_models']
                if 'path_styles' in paths_obj:
                    paths_dict['styles_path'] = paths_obj['path_styles']
                if 'path_wildcards' in paths_obj:
                    paths_dict['wildcards_path'] = paths_obj['path_wildcards']
                if 'path_outputs' in paths_obj:
                    paths_dict['temp_outputs_path'] = paths_obj['path_outputs']

            except Exception as e:
                print('load_paths, e: ' + str(e))
            finally:
                paths_file.close()

    return paths_dict


config_path_mre = "paths.json"
config_path = "user_path_config.txt"
config_dict = {}


try:
    if os.path.exists(config_path_mre):
        with open(config_path_mre, "r", encoding="utf-8") as json_file:
            config_dict = load_paths(config_path_mre)
    elif os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict = json.load(json_file)
except Exception as e:
    print('Load path config failed')
    print(e)


def get_config_or_set_default(key, default):
    global config_dict
    v = config_dict.get(key, None)
    if not isinstance(v, str):
        v = default
    dp = v if os.path.isabs(v) else os.path.abspath(os.path.join(os.path.dirname(__file__), v))
    if not os.path.exists(dp) or not os.path.isdir(dp):
        os.makedirs(dp, exist_ok=True)
    config_dict[key] = dp
    return dp


modelfile_path = get_config_or_set_default('modelfile_path', '../models/checkpoints/')
lorafile_path = get_config_or_set_default('lorafile_path', '../models/loras/')
embeddings_path = get_config_or_set_default('embeddings_path', '../models/embeddings/')
clip_vision_path = get_config_or_set_default('clip_vision_path', '../models/clip_vision/')
controlnet_path = get_config_or_set_default('controlnet_path', '../models/controlnet/')
styles_path = get_config_or_set_default('styles_path', '../sdxl_styles/')
wildcards_path = get_config_or_set_default('wildcards_path', '../wildcards/')
vae_approx_path = get_config_or_set_default('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_config_or_set_default('upscale_models_path', '../models/upscale_models/')
inpaint_models_path = get_config_or_set_default('inpaint_models_path', '../models/inpaint/')
fooocus_expansion_path = get_config_or_set_default('fooocus_expansion_path',
                                                   '../models/prompt_expansion/fooocus_expansion')

temp_outputs_path = get_config_or_set_default('temp_outputs_path', '../outputs/')
last_prompt_path = os.path.join(temp_outputs_path, 'last_prompt.json')

with open(config_path, "w", encoding="utf-8") as json_file:
    json.dump(config_dict, json_file, indent=4)


os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora_name = 'sd_xl_offset_example-lora_1.0.safetensors'
default_clip_vision_name = 'clip_vision_g.safetensors'
default_controlnet_canny_name = 'control-lora-canny-rank128.safetensors'
default_controlnet_depth_name = 'control-lora-depth-rank128.safetensors'
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []
canny_filenames = []
depth_filenames = []


def get_files_from_folder(folder_path, exensions=None, name_filter=None):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in files:
            _, file_extension = os.path.splitext(filename)
            if (exensions == None or file_extension.lower() in exensions) and (name_filter == None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return sorted(filenames, key=lambda x: -1 if os.sep in x else 1)


def get_model_filenames(folder_path, name_filter=None):
    return get_files_from_folder(folder_path, ['.pth', '.ckpt', '.bin', '.safetensors'], name_filter)


def update_all_model_names():
    global model_filenames, lora_filenames, canny_filenames, depth_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    canny_filenames = get_model_filenames(controlnet_path, 'control-lora-canny')
    depth_filenames = get_model_filenames(controlnet_path, 'control-lora-depth')
    return


def downloading_inpaint_models():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=inpaint_models_path,
        file_name='fooocus_inpaint_head.pth'
    )
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
        model_dir=inpaint_models_path,
        file_name='inpaint.fooocus.patch'
    )
    return os.path.join(inpaint_models_path, 'fooocus_inpaint_head.pth'), os.path.join(inpaint_models_path, 'inpaint.fooocus.patch')


update_all_model_names()
