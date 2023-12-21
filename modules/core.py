from modules.patch import patch_all

patch_all()


import os
import random
import einops
import torch
import numpy as np

import comfy.model_management
import comfy.utils

from comfy.sd import load_checkpoint_guess_config
from nodes import VAEDecode, EmptyLatentImage, CLIPTextEncode, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, VAEEncodeForInpaint, \
    ConditioningZeroOut, ConditioningAverage, CLIPVisionEncode, unCLIPConditioning, ControlNetApplyAdvanced
from comfy.sample import prepare_mask, broadcast_cond, get_additional_models, cleanup_additional_models
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_canny import Canny
from comfy_extras.nodes_freelunch import FreeU
from comfy.model_base import SDXL, SDXLRefiner
from comfy.lora import model_lora_keys_unet, model_lora_keys_clip, load_lora
from modules.samplers_advanced import KSamplerBasic, KSamplerWithRefiner
from modules.path import embeddings_path


opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opVAEEncodeForInpaint = VAEEncodeForInpaint()
opImageScaleToTotalPixels = ImageScaleToTotalPixels()
opConditioningZeroOut = ConditioningZeroOut()
opConditioningAverage = ConditioningAverage()
opCLIPVisionEncode = CLIPVisionEncode()
opUnCLIPConditioning = unCLIPConditioning()
opCanny = Canny()
opFreeU = FreeU()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()


class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision, model_filename=None):
        if isinstance(model_filename, str):
            is_refiner = isinstance(unet.model, SDXLRefiner)
            if unet is not None:
                unet.model.model_file = dict(filename=model_filename, prefix='model')
            if clip is not None:
                clip.cond_stage_model.model_file = dict(filename=model_filename, prefix='refiner_clip' if is_refiner else 'base_clip')
            if vae is not None:
                vae.first_stage_model.model_file = dict(filename=model_filename, prefix='first_stage_model')
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=embeddings_path)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, model_filename=ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def load_sd_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora = comfy.utils.load_torch_file(lora_filename, safe_load=False)

    if lora_filename.lower().endswith('.fooocus.patch'):
        loaded = lora
    else:
        key_map = model_lora_keys_unet(model.unet.model)
        key_map = model_lora_keys_clip(model.clip.cond_stage_model, key_map)
        loaded = load_lora(lora, key_map)

    new_modelpatcher = model.unet.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)

    new_clip = model.clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)

    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("Lora missed: ", x)

    unet, clip = new_modelpatcher, new_clip
    return StableDiffusionModel(unet=unet, clip=clip, vae=model.vae, clip_vision=model.clip_vision)


@torch.no_grad()
@torch.inference_mode()
def load_clip_vision(ckpt_filename):
    return comfy.clip_vision.load(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return comfy.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def encode_prompt_condition(clip, prompt):
    return opCLIPTextEncode.encode(clip=clip, text=prompt)[0]


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    if tiled:
        return opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
    else:
        return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels):
    return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def upscale(image, megapixels=1.0):
    return opImageScaleToTotalPixels.upscale(image=image, upscale_method='bicubic', megapixels=megapixels)[0]


@torch.no_grad()
@torch.inference_mode()
def zero_out(conditioning):
    return opConditioningZeroOut.zero_out(conditioning=conditioning)[0]


@torch.no_grad()
@torch.inference_mode()
def average(conditioning_to, conditioning_from, conditioning_to_strength):
    return opConditioningAverage.addWeighted(conditioning_to=conditioning_to, conditioning_from=conditioning_from, conditioning_to_strength=conditioning_to_strength)[0]


@torch.no_grad()
@torch.inference_mode()
def set_conditioning_strength(conditioning, strength):
    return average(conditioning, zero_out(conditioning), strength)


@torch.no_grad()
@torch.inference_mode()
def encode_clip_vision(clip_vision, image):
    return opCLIPVisionEncode.encode(clip_vision=clip_vision, image=image)[0]


@torch.no_grad()
@torch.inference_mode()
def apply_adm(conditioning, clip_vision_output, strength, noise_augmentation):
    return opUnCLIPConditioning.apply_adm(conditioning=conditioning, clip_vision_output=clip_vision_output, strength=strength, noise_augmentation=noise_augmentation)[0]


@torch.no_grad()
@torch.inference_mode()
def detect_edge(image, low_threshold, high_threshold):
    return opCanny.detect_edge(image=image, low_threshold=low_threshold, high_threshold=high_threshold)[0]


@torch.no_grad()
@torch.inference_mode()
def freeu(model, b1, b2, s1, s2):
    unet = opFreeU.patch(model=model.unet, b1=b1, b2=b2, s1=s1, s2=s2)[0]
    return StableDiffusionModel(unet=unet, clip=model.clip, vae=model.vae, clip_vision=model.clip_vision)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
        image=image, strength=strength, start_percent=start_percent, end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if tiled:
        return opVAEEncodeTiled.encode(pixels=pixels, vae=vae, tile_size=512)[0]
    else:
        return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    return opVAEEncodeForInpaint.encode(pixels=pixels, vae=vae, mask=mask)[0]


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_model = None
taesd = None


@torch.no_grad()
@torch.inference_mode()
def get_previewer(device, latent_format, is_sdxl=True):
    global VAE_approx_model, taesd

    if VAE_approx_model is None and is_sdxl:
        from modules.path import vae_approx_path
        vae_approx_filename = os.path.join(vae_approx_path, 'xlvaeapp.pth')
        sd = torch.load(vae_approx_filename, map_location='cpu')
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if comfy.model_management.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(comfy.model_management.get_torch_device())

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    if taesd is None and not is_sdxl:
        from latent_preview import TAESD, TAESDPreviewerImpl
        from modules.path import vae_approx_path
        taesd_decoder_path = os.path.join(vae_approx_path, latent_format.taesd_decoder_name)

        if not os.path.exists(taesd_decoder_path):
            print(f"Warning: TAESD previews enabled, but could not find {taesd_decoder_path}")
            return None

        taesd = TAESD(None, taesd_decoder_path).to(device)

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function_sd(x0, step, total_steps):
        with torch.no_grad():
            x_sample = taesd.decoder(torch.nn.functional.avg_pool2d(x0, kernel_size=(2, 2))).detach() * 255.0
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample[0]

    return preview_function if is_sdxl else preview_function_sd

@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_fooocus_2m_sde_inpaint_seamless',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(0, 2**63 - 1)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format, isinstance(model.model, SDXL))

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer is not None:
            y = previewer(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    real_model = None
    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, comfy.model_management.batch_area_memory(noise.shape[0] * noise.shape[2] * noise.shape[3]) + inference_memory)
    real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    sampler = KSamplerBasic(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler,
                       denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
@torch.inference_mode()
def ksampler_with_refiner(model, positive, negative, refiner, refiner_positive, refiner_negative, latent,
                          seed=None, steps=30, refiner_switch_step=20, cfg=7.0, sampler_name='dpmpp_fooocus_2m_sde_inpaint_seamless',
                          scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, callback_function=None):
    # SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    # SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    #             "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
    #             "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "ddim", "uni_pc", "uni_pc_bh2"]

    seed = seed if isinstance(seed, int) else random.randint(0, 2**63 - 1)

    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(device, model.model.latent_format, isinstance(model.model, SDXL))

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        y = None
        if previewer is not None:
            y = previewer(x0, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, y)
        pbar.update_absolute(step + 1, total_steps, None)

    sigmas = None
    disable_pbar = False

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, comfy.model_management.batch_area_memory(noise.shape[0] * noise.shape[2] * noise.shape[3]) + inference_memory)

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    refiner_positive_copy = broadcast_cond(refiner_positive, noise.shape[0], device)
    refiner_negative_copy = broadcast_cond(refiner_negative, noise.shape[0], device)

    sampler = KSamplerWithRefiner(model=model, refiner_model=refiner, steps=steps, device=device,
                                  sampler=sampler_name, scheduler=scheduler,
                                  denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, refiner_positive=refiner_positive_copy,
                             refiner_negative=refiner_negative_copy, refiner_switch_step=refiner_switch_step,
                             cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback_function=callback, disable_pbar=disable_pbar,
                             seed=seed)

    samples = samples.cpu()

    cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples

    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
