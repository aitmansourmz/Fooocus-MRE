import modules.core as core
import os
import gc
import torch
import numpy as np
import modules.path
import modules.virtual_memory as virtual_memory
import comfy.model_management

from comfy.model_base import BaseModel, SDXL, SDXLRefiner
from modules.settings import default_settings
from modules.patch import set_comfy_adm_encoding, set_fooocus_adm_encoding, cfg_patched, patched_model_function
from modules.expansion import FooocusExpansion


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''

clip_vision: core.StableDiffusionModel = None
clip_vision_hash = ''

controlnet_canny: core.StableDiffusionModel = None
controlnet_canny_hash = ''

controlnet_depth: core.StableDiffusionModel = None
controlnet_depth_hash = ''


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.path.modelfile_path, name)))
    model_hash = filename

    if xl_base_hash == model_hash:
        return

    if xl_base is not None:
        xl_base = None

    if xl_base_patched is not None:
        xl_base_patched = None

    xl_base = core.load_model(filename)
    if not isinstance(xl_base.unet.model, BaseModel):
        print(f'Model not supported: {name}, using default base model instead.')
        xl_base = None
        xl_base_hash = ''
        refresh_base_model(modules.path.default_base_model_name)
        xl_base_hash = model_hash
        xl_base_patched = xl_base
        xl_base_patched_hash = ''
        return

    if not isinstance(xl_base.unet.model, SDXL):
        print('WARNING: loading non-SDXL base model.')

    xl_base_hash = model_hash
    xl_base_patched = xl_base
    xl_base_patched_hash = ''
    print(f'Base model loaded: {model_hash}')
    return


def is_base_sdxl():
    assert xl_base is not None
    return isinstance(xl_base.unet.model, SDXL)


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name):
    global xl_refiner, xl_refiner_hash

    filename = os.path.abspath(os.path.realpath(os.path.join(modules.path.modelfile_path, name)))
    model_hash = filename

    if xl_refiner_hash == model_hash:
        return

    if name == 'None':
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    if xl_refiner is not None:
        xl_refiner = None

    xl_refiner = core.load_model(filename)
    if not isinstance(xl_refiner.unet.model, SDXLRefiner):
        print('Model not supported. Fooocus only support SDXL refiner as the refiner.')
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    xl_refiner_hash = model_hash
    print(f'Refiner model loaded: {model_hash}')

    xl_refiner.vae = None
    return


@torch.no_grad()
@torch.inference_mode()
def patch_base(loras, freeu, b1, b2, s1, s2):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras + [freeu, b1, b2, s1, s2]):
        return

    model = xl_base
    for name, weight in loras:
        if name == 'None':
            continue

        if os.path.exists(name):
            filename = name
        else:
            filename = os.path.join(modules.path.lorafile_path, name)

        assert os.path.exists(filename), 'Lora file not found!'

        model = core.load_sd_lora(model, filename, strength_model=weight, strength_clip=weight)
    if freeu:
        xl_base_patched = core.freeu(model, b1, b2, s1, s2)
    else:
        xl_base_patched = model
    xl_base_patched_hash = str(loras + [freeu, b1, b2, s1, s2])
    print(f'LoRAs loaded: {loras}')
    if freeu:
        print(f'FreeU applied: {[b1, b2, s1, s2]}')

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_clip_vision():
    global clip_vision, clip_vision_hash
    if clip_vision_hash == str(clip_vision):
        return

    model_name = modules.path.default_clip_vision_name
    filename = os.path.join(modules.path.clip_vision_path, model_name)
    clip_vision = core.load_clip_vision(filename)

    clip_vision_hash = model_name
    print(f'CLIP Vision model loaded: {clip_vision_hash}')

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnet_canny(name=None):
    global controlnet_canny, controlnet_canny_hash
    if controlnet_canny_hash == str(controlnet_canny):
        return

    model_name = modules.path.default_controlnet_canny_name if name == None else name
    filename = os.path.join(modules.path.controlnet_path, model_name)
    controlnet_canny = core.load_controlnet(filename)

    controlnet_canny_hash = model_name
    print(f'ControlNet model loaded: {controlnet_canny_hash}')

    return



@torch.no_grad()
@torch.inference_mode()
def refresh_controlnet_depth(name=None):
    global controlnet_depth, controlnet_depth_hash
    if controlnet_depth_hash == str(controlnet_depth):
        return

    model_name = modules.path.default_controlnet_depth_name if name == None else name
    filename = os.path.join(modules.path.controlnet_path, model_name)
    controlnet_depth = core.load_controlnet(filename)

    controlnet_depth_hash = model_name
    print(f'ControlNet model loaded: {controlnet_depth_hash}')

    return


@torch.no_grad()
@torch.inference_mode()
def set_clip_skips(base_clip_skip, refiner_clip_skip):
    xl_base_patched.clip.clip_layer(base_clip_skip)
    if xl_refiner is not None:
        xl_refiner.clip.clip_layer(refiner_clip_skip)
    return


@torch.no_grad()
@torch.inference_mode()
def apply_prompt_strength(base_cond, refiner_cond, prompt_strength=1.0):
    if prompt_strength >= 0 and prompt_strength < 1.0:
        base_cond = core.set_conditioning_strength(base_cond, prompt_strength)

    if xl_refiner is not None:
        if prompt_strength >= 0 and prompt_strength < 1.0:
            refiner_cond = core.set_conditioning_strength(refiner_cond, prompt_strength)
    else:
        refiner_cond = None
    return base_cond, refiner_cond


@torch.no_grad()
@torch.inference_mode()
def apply_revision(base_cond, revision=False, revision_strengths=[], clip_vision_outputs=[]):
    if revision:
        set_comfy_adm_encoding()
        for i in range(len(clip_vision_outputs)):
            if revision_strengths[i % 4] != 0:
                base_cond = core.apply_adm(base_cond, clip_vision_outputs[i % 4], revision_strengths[i % 4], 0)
    else:
        set_fooocus_adm_encoding()
    return base_cond


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[text] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clip_encode(sd, texts, pool_top_k=1):
    if sd is None:
        return None
    if sd.clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    clip = sd.clip
    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def clear_sd_cond_cache(sd):
    if sd is None:
        return None
    if sd.clip is None:
        return None
    sd.clip.fcs_cond_cache = {}
    return


@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    clear_sd_cond_cache(xl_base_patched)
    clear_sd_cond_cache(xl_refiner)
    gc.collect()
    comfy.model_management.soft_empty_cache()


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras, freeu, b1, b2, s1, s2):
    refresh_refiner_model(refiner_model_name)
    if xl_refiner is not None:
        virtual_memory.try_move_to_virtual_memory(xl_refiner.unet.model)
        virtual_memory.try_move_to_virtual_memory(xl_refiner.clip.cond_stage_model)

    refresh_base_model(base_model_name)
    virtual_memory.load_from_virtual_memory(xl_base.unet.model)

    patch_base(loras, freeu, b1, b2, s1, s2)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=default_settings['refiner_model'],
    base_model_name=default_settings['base_model'],
    loras=[(default_settings['lora_1_model'], default_settings['lora_1_weight']),
        (default_settings['lora_2_model'], default_settings['lora_2_weight']),
        (default_settings['lora_3_model'], default_settings['lora_3_weight']),
        (default_settings['lora_4_model'], default_settings['lora_4_weight']),
        (default_settings['lora_5_model'], default_settings['lora_5_weight'])],
    freeu=default_settings['freeu'],
    b1=default_settings['freeu_b1'],
    b2=default_settings['freeu_b2'],
    s1=default_settings['freeu_s1'],
    s2=default_settings['freeu_s2']
)

expansion = FooocusExpansion()


@torch.no_grad()
@torch.inference_mode()
def patch_all_models():
    assert xl_base is not None
    assert xl_base_patched is not None

    xl_base.unet.model_options['sampler_cfg_function'] = cfg_patched
    xl_base.unet.model_options['model_function_wrapper'] = patched_model_function

    xl_base_patched.unet.model_options['sampler_cfg_function'] = cfg_patched
    xl_base_patched.unet.model_options['model_function_wrapper'] = patched_model_function

    if xl_refiner is not None:
        xl_refiner.unet.model_options['sampler_cfg_function'] = cfg_patched
        xl_refiner.unet.model_options['model_function_wrapper'] = patched_model_function

    return


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, sampler_name, scheduler, cfg, img2img, input_image, start_step,
        control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength,
        control_lora_depth, depth_start, depth_stop, depth_strength, callback, latent=None, denoise=1.0, tiled=False):

    patch_all_models()

    if xl_refiner is not None:
        virtual_memory.try_move_to_virtual_memory(xl_refiner.unet.model)
    virtual_memory.load_from_virtual_memory(xl_base.unet.model)

    if img2img and input_image != None:
        initial_latent = core.encode_vae(vae=xl_base_patched.vae, pixels=input_image)
        force_full_denoise = False
    elif latent is None:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        force_full_denoise = True
    else:
        initial_latent = latent
        force_full_denoise = False

    positive_conditions = positive_cond[0]
    negative_conditions = negative_cond[0]

    if control_lora_canny and input_image != None:
        edges_image = core.detect_edge(input_image, canny_edge_low, canny_edge_high)
        positive_conditions, negative_conditions = core.apply_controlnet(positive_conditions, negative_conditions,
            controlnet_canny, edges_image, canny_strength, canny_start, canny_stop)

    if control_lora_depth and input_image != None:
        positive_conditions, negative_conditions = core.apply_controlnet(positive_conditions, negative_conditions,
            controlnet_depth, input_image, depth_strength, depth_start, depth_stop)

    if xl_refiner is not None and is_base_sdxl():
        positive_conditions_refiner = positive_cond[1]
        negative_conditions_refiner = negative_cond[1]

        sampled_latent = core.ksampler_with_refiner(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            refiner=xl_refiner.unet,
            refiner_positive=positive_conditions_refiner,
            refiner_negative=negative_conditions_refiner,
            refiner_switch_step=switch,
            latent=initial_latent,
            steps=steps, start_step=start_step, last_step=steps,
            disable_noise=False, force_full_denoise=force_full_denoise, denoise=denoise,
            seed=image_seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cfg=cfg,
            callback_function=callback
        )
    else:
        sampled_latent = core.ksampler(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            latent=initial_latent,
            steps=steps, start_step=start_step, last_step=steps,
            disable_noise=False, force_full_denoise=force_full_denoise, denoise=denoise,
            seed=image_seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cfg=cfg,
            callback_function=callback
        )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent, tiled=tiled)
    images = core.pytorch_to_numpy(decoded_latent)

    return images
