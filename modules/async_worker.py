import threading


buffer = []
outputs = []


def worker():
    global buffer, outputs

    import os
    import json
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.path
    import modules.patch
    import fooocus_version
    import modules.virtual_memory as virtual_memory
    import comfy.model_management
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants

    from PIL import Image, ImageOps
    from modules.settings import default_settings
    from modules.resolutions import annotate_resolution_string, get_resolution_string, resolutions, string_to_dimensions
    from modules.sdxl_styles import apply_style, apply_wildcards, style_keys
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import join_prompts, remove_empty_str, HWC3, resize_image, image_is_generated_in_current_ui
    from modules.upscaler import perform_upscale

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)


    def get_image(path, megapixels=1.0):
        image = None
        with open(path, 'rb') as image_file:
            pil_image = Image.open(image_file)
            image = ImageOps.exif_transpose(pil_image)
            image_file.close()
            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image = core.upscale(image, megapixels)
        return image


    def progressbar(number, text):
        print(f'[Fooocus] {text}')
        outputs.append(['preview', (number, text, None)])


    @torch.no_grad()
    @torch.inference_mode()
    def handler(task):
        prompt, negative_prompt, style_selections, performance, resolution, image_number, image_seed, \
        sharpness, sampler_name, scheduler, custom_steps, custom_switch, cfg, \
        base_model_name, refiner_model_name, base_clip_skip, refiner_clip_skip, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, \
        save_metadata_json, save_metadata_image, \
        img2img_mode, img2img_start_step, img2img_denoise, img2img_scale, \
        revision_mode, positive_prompt_strength, negative_prompt_strength, revision_strength_1, revision_strength_2, \
        revision_strength_3, revision_strength_4, same_seed_for_all, output_format, \
        control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength, canny_model, \
        control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, use_expansion, \
        freeu, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
        input_image_checkbox, current_tab, \
        uov_method, uov_input_image, outpaint_selections, inpaint_input_image, \
        use_style_iterator, input_gallery, revision_gallery, keep_input_names = task

        outpaint_selections = [o.lower() for o in outpaint_selections]

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        loras_user_raw_input = copy.deepcopy(loras)

        if use_style_iterator:
            style_iterator_pool = style_keys.copy()
            for s in style_selections:
                style_iterator_pool.remove(s)

        uov_method = uov_method.lower()

        modules.patch.sharpness = sharpness
        modules.patch.negative_adm = True
        initial_latent = None
        denoising_strength = 1.0
        tiled = False
        inpaint_worker.current_task = None


        if performance == 'Speed':
            steps = constants.STEPS_SPEED
            switch = constants.SWITCH_SPEED
        elif performance == 'Quality':
            steps = constants.STEPS_QUALITY
            switch = constants.SWITCH_QUALITY
        else:
            steps = custom_steps
            switch = round(custom_steps * custom_switch)


        pipeline.clear_all_caches()  # save memory


        if resolution not in resolutions:
            try:
                resolution = annotate_resolution_string(resolution)
            except Exception as e:
                print(f'Problem with resolution definition: "{resolution}", reverting to default: ' + default_settings['resolution'])
                resolution = default_settings['resolution']
        width, height = string_to_dimensions(resolution)


        if input_image_checkbox:
            progressbar(0, 'Image processing ...')
            if current_tab == 'uov' and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    if not image_is_generated_in_current_ui(uov_input_image, ui_width=width, ui_height=height):
                        uov_input_image = resize_image(uov_input_image, width=width, height=height)
                        print(f'Resolution corrected - users are uploading their own images.')
                    else:
                        print(f'Processing images generated by Fooocus.')
                    if 'subtle' in uov_method:
                        denoising_strength = 0.5
                    if 'strong' in uov_method:
                        denoising_strength = 0.85
                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(0, 'VAE encoding ...')
                    initial_latent = core.encode_vae(vae=pipeline.xl_base_patched.vae, pixels=initial_pixels)
                    B, C, H, W = initial_latent['samples'].shape
                    width = W * 8
                    height = H * 8
                    print(f'Final resolution is {str((height, width))}.')
                elif 'upscale' in uov_method:
                    H, W, C = uov_input_image.shape
                    progressbar(0, f'Upscaling image from {str((H, W))} ...')

                    uov_input_image = core.numpy_to_pytorch(uov_input_image)
                    uov_input_image = perform_upscale(uov_input_image)
                    uov_input_image = core.pytorch_to_numpy(uov_input_image)[0]
                    print(f'Image upscaled.')

                    if '1.5x' in uov_method:
                        f = 1.5
                    elif '2x' in uov_method:
                        f = 2.0
                    else:
                        f = 1.0

                    width_f = int(width * f)
                    height_f = int(height * f)

                    if image_is_generated_in_current_ui(uov_input_image, ui_width=width_f, ui_height=height_f):
                        uov_input_image = resize_image(uov_input_image, width=int(W * f), height=int(H * f))
                        print(f'Processing images generated by Fooocus.')
                    else:
                        uov_input_image = resize_image(uov_input_image, width=width_f, height=height_f)
                        print(f'Resolution corrected - users are uploading their own images.')

                    H, W, C = uov_input_image.shape
                    image_is_super_large = H * W > 2800 * 2800

                    if 'fast' in uov_method:
                        direct_return = True
                    elif image_is_super_large:
                        print('Image is too large. Directly returned the SR image. '
                              'Usually directly return SR image at 4K resolution '
                              'yields better results than SDXL diffusion.')
                        direct_return = True
                    else:
                        direct_return = False

                    if direct_return:
                        d = [('Upscale (Fast)', '2x')]
                        log(uov_input_image, d, single_line_number=1, output_format=output_format)
                        outputs.append(['results', [uov_input_image]])
                        return

                    tiled = True
                    denoising_strength = 1.0 - 0.618
                    steps = int(steps * 0.618)
                    switch = int(steps * 0.67)
                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(0, 'VAE encoding ...')

                    initial_latent = core.encode_vae(vae=pipeline.xl_base_patched.vae, pixels=initial_pixels, tiled=True)
                    B, C, H, W = initial_latent['samples'].shape
                    width = W * 8
                    height = H * 8
                    print(f'Final resolution is {str((height, width))}.')
            if current_tab == 'inpaint' and isinstance(inpaint_input_image, dict):
                sampler_name = 'dpmpp_fooocus_2m_sde_inpaint_seamless'
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    if len(outpaint_selections) > 0:
                        H, W, C = inpaint_image.shape
                        if 'top' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant', constant_values=255)
                        if 'bottom' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant', constant_values=255)

                        H, W, C = inpaint_image.shape
                        if 'left' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant', constant_values=255)
                        if 'right' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant', constant_values=255)

                        inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                        inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())

                    inpaint_worker.current_task = inpaint_worker.InpaintWorker(image=inpaint_image, mask=inpaint_mask,
                                                                               is_outpaint=len(outpaint_selections) > 0)

                    # print(f'Inpaint task: {str((height, width))}')
                    # outputs.append(['results', inpaint_worker.current_task.visualize_mask_processing()])
                    # return

                    progressbar(0, 'Downloading inpainter ...')
                    inpaint_head_model_path, inpaint_patch_model_path = modules.path.downloading_inpaint_models()
                    loras += [(inpaint_patch_model_path, 1.0)]

                    inpaint_pixels = core.numpy_to_pytorch(inpaint_worker.current_task.image_ready)
                    progressbar(0, 'VAE encoding ...')
                    initial_latent = core.encode_vae(vae=pipeline.xl_base_patched.vae, pixels=inpaint_pixels)
                    inpaint_latent = initial_latent['samples']
                    B, C, H, W = inpaint_latent.shape
                    inpaint_mask = core.numpy_to_pytorch(inpaint_worker.current_task.mask_ready[None])
                    inpaint_mask = torch.nn.functional.avg_pool2d(inpaint_mask, (8, 8))
                    inpaint_mask = torch.nn.functional.interpolate(inpaint_mask, (H, W), mode='bilinear')
                    inpaint_worker.current_task.load_latent(latent=inpaint_latent, mask=inpaint_mask)

                    progressbar(0, 'VAE inpaint encoding ...')

                    inpaint_mask = (inpaint_worker.current_task.mask_ready > 0).astype(np.float32)
                    inpaint_mask = torch.tensor(inpaint_mask).float()

                    vae_dict = core.encode_vae_inpaint(
                        mask=inpaint_mask, vae=pipeline.xl_base_patched.vae, pixels=inpaint_pixels)

                    inpaint_latent = vae_dict['samples']
                    inpaint_mask = vae_dict['noise_mask']
                    inpaint_worker.current_task.load_inpaint_guidance(latent=inpaint_latent, mask=inpaint_mask, model_path=inpaint_head_model_path)

                    B, C, H, W = inpaint_latent.shape
                    height, width = inpaint_worker.current_task.image_raw.shape[:2]
                    print(f'Final resolution is {str((height, width))}, latent is {str((H * 8, W * 8))}.')


        input_gallery_size = len(input_gallery)
        if input_gallery_size == 0:
            img2img_mode = False
            input_image_path = None
            control_lora_canny = False
            control_lora_depth = False

        revision_gallery_size = len(revision_gallery)
        if revision_gallery_size == 0:
            revision_mode = False


        progressbar(1, 'Initializing ...')

        raw_prompt = prompt
        raw_negative_prompt = negative_prompt

        prompts = remove_empty_str([safe_str(p) for p in prompt.split('\n')], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.split('\n')], default='')

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

        try:
            seed = int(image_seed) 
        except Exception as e:
            seed = -1
        if not isinstance(seed, int) or seed < constants.MIN_SEED or seed > constants.MAX_SEED:
            seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)


        progressbar(3, 'Loading models ...')
        pipeline.refresh_everything(
            refiner_model_name=refiner_model_name,
            base_model_name=base_model_name,
            loras=loras,
            freeu=freeu,
            b1=freeu_b1,
            b2=freeu_b2,
            s1=freeu_s1,
            s2=freeu_s2)

        is_sdxl = pipeline.is_base_sdxl()
        if not is_sdxl:
            print('WARNING: using non-SDXL base model (supported in limited scope).')
            control_lora_canny = False
            control_lora_depth = False
            revision_mode = False

        pipeline.set_clip_skips(base_clip_skip, refiner_clip_skip)
        if revision_mode:
            pipeline.refresh_clip_vision()
        if control_lora_canny:
            pipeline.refresh_controlnet_canny(canny_model)
        if control_lora_depth:
            pipeline.refresh_controlnet_depth(depth_model)


        clip_vision_outputs = []
        if revision_mode:
            revision_images_paths = list(map(lambda x: x['name'], revision_gallery))
            revision_images_filenames = list(map(lambda path: os.path.basename(path), revision_images_paths))
            revision_strengths = [revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4]
            for i in range(revision_gallery_size):
                progressbar(4, f'Revision for image {i + 1} ...')
                print(f'Revision for image {i+1} started')
                if revision_strengths[i % 4] != 0:
                    revision_image = get_image(revision_images_paths[i])
                    clip_vision_output = core.encode_clip_vision(pipeline.clip_vision, revision_image)
                    clip_vision_outputs.append(clip_vision_output)
                print(f'Revision for image {i+1} finished')
        else:
            revision_images_paths = []
            revision_images_filenames = []
            revision_strengths = []


        progressbar(5, 'Processing prompts ...')
        tasks = []
        for i in range(image_number):
            positive_basic_workloads = []
            negative_basic_workloads = []
            task_seed = seed if same_seed_for_all else seed + i
            task_prompt = apply_wildcards(prompt, task_seed)

            if use_style_iterator and i > 0: # original styles selection goes as 0th image
                task_style_selections = style_selections + [style_iterator_pool[i - 1]]
            else:
                task_style_selections = style_selections

            use_style = len(task_style_selections) > 0

            if use_style:
                for s in task_style_selections:
                    p, n = apply_style(s, positive=task_prompt)
                    positive_basic_workloads.append(p)
                    negative_basic_workloads.append(n)
            else:
                positive_basic_workloads.append(task_prompt)
    
            negative_basic_workloads.append(negative_prompt)  # Always use independent workload for negative.

            positive_basic_workloads = positive_basic_workloads + extra_positive_prompts
            negative_basic_workloads = negative_basic_workloads + extra_negative_prompts

            positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=negative_prompt)

            tasks.append(dict(
                task_seed=task_seed,
                prompt=task_prompt,
                style_selections=task_style_selections,
                positive=positive_basic_workloads,
                negative=negative_basic_workloads,
                positive_top_k=len(positive_basic_workloads),
                negative_top_k=len(negative_basic_workloads),
                expansion='',
                c=[None, None],
                uc=[None, None]
            ))


        if use_expansion:
            for i, t in enumerate(tasks):
                progressbar(5, f'Preparing Fooocus text #{i + 1} ...')
                expansion = pipeline.expansion(t['prompt'], t['task_seed'])
                print(f'[Prompt Expansion] New suffix: {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(t['prompt'], expansion)]  # Deep copy.

        for i, t in enumerate(tasks):
            progressbar(7, f'Encoding base positive #{i + 1} ...')
            t['c'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['positive'],
                                             pool_top_k=t['positive_top_k'])

        for i, t in enumerate(tasks):
            progressbar(9, f'Encoding base negative #{i + 1} ...')
            t['uc'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['negative'],
                                              pool_top_k=t['negative_top_k'])

        if pipeline.xl_refiner is not None:
            virtual_memory.load_from_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)

            for i, t in enumerate(tasks):
                progressbar(11, f'Encoding refiner positive #{i + 1} ...')
                t['c'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['positive'],
                                                 pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                progressbar(13, f'Encoding refiner negative #{i + 1} ...')
                t['uc'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['negative'],
                                                  pool_top_k=t['negative_top_k'])

            virtual_memory.try_move_to_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)


        for i, t in enumerate(tasks):
            progressbar(13, f'Applying prompt strengths #{i + 1} ...')
            t['c'][0], t['c'][1] = pipeline.apply_prompt_strength(t['c'][0], t['c'][1], positive_prompt_strength)
            t['uc'][0], t['uc'][1] = pipeline.apply_prompt_strength(t['uc'][0], t['uc'][1], negative_prompt_strength)

        for i, t in enumerate(tasks):
            progressbar(13, f'Applying Revision #{i + 1} ...')
            t['c'][0] = pipeline.apply_revision(t['c'][0], revision_mode, revision_strengths, clip_vision_outputs)


        pipeline.clear_all_caches()  # save memory

        results = []
        metadata_strings = []
        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            comfy.model_management.throw_exception_if_processing_interrupted()
            done_steps = current_task_idx * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_idx + 1}-th Sampling',
                y)])

        print(f'[ADM] Negative ADM = {modules.patch.negative_adm}')

        outputs.append(['preview', (13, 'Starting tasks ...', None)])
        for current_task_idx, task in enumerate(tasks):
            if img2img_mode or control_lora_canny or control_lora_depth:
                input_gallery_entry = input_gallery[current_task_idx % input_gallery_size]
                input_image_path = input_gallery_entry['name']
                input_image_filename = None if input_image_path == None else os.path.basename(input_image_path)
            else:
                input_image_path = None
                input_image_filename = None
                keep_input_names = None

            if img2img_mode:
                start_step = round(steps * img2img_start_step)
                denoise = img2img_denoise
            else:
                start_step = 0
                denoise = denoising_strength

            input_image = None
            if input_image_path != None:
                img2img_megapixels = width * height * img2img_scale ** 2 / 2**20
                min_mp = constants.MIN_MEGAPIXELS if is_sdxl else constants.MIN_MEGAPIXELS_SD
                max_mp = constants.MAX_MEGAPIXELS if is_sdxl else constants.MAX_MEGAPIXELS_SD
                if img2img_megapixels < min_mp:
                    img2img_megapixels = min_mp
                elif img2img_megapixels > max_mp:
                    img2img_megapixels = max_mp
                input_image = get_image(input_image_path, img2img_megapixels)

            try:
                execution_start_time = time.perf_counter()

                imgs = pipeline.process_diffusion(
                    positive_cond=task['c'],
                    negative_cond=task['uc'],
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    cfg=cfg,
                    img2img=img2img_mode, #? -> latent
                    input_image=input_image, #? -> latent
                    start_step=start_step,
                    control_lora_canny=control_lora_canny,
                    canny_edge_low=canny_edge_low,
                    canny_edge_high=canny_edge_high,
                    canny_start=canny_start,
                    canny_stop=canny_stop,
                    canny_strength=canny_strength,
                    control_lora_depth=control_lora_depth,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    depth_strength=depth_strength,
                    callback=callback,
                    latent=initial_latent,
                    denoise=denoise,
                    tiled=tiled)

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                execution_time = time.perf_counter() - execution_start_time
                print(f'Diffusion time: {execution_time:.2f} seconds')
    
                metadata = {
                    'prompt': raw_prompt, 'negative_prompt': raw_negative_prompt, 'styles': task['style_selections'],
                    'real_prompt': task['positive'], 'real_negative_prompt': task['negative'],
                    'seed': task['task_seed'], 'width': width, 'height': height,
                    'sampler': sampler_name, 'scheduler': scheduler, 'performance': performance,
                    'steps': steps, 'switch': switch, 'sharpness': sharpness, 'cfg': cfg,
                    'base_clip_skip': base_clip_skip, 'refiner_clip_skip': refiner_clip_skip,
                    'base_model': base_model_name, 'refiner_model': refiner_model_name,
                    'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                    'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5, 'freeu': freeu,
                    'img2img': img2img_mode, 'revision': revision_mode,
                    'positive_prompt_strength': positive_prompt_strength, 'negative_prompt_strength': negative_prompt_strength,
                    'control_lora_canny': control_lora_canny, 'control_lora_depth': control_lora_depth,
                    'prompt_expansion': use_expansion
                }
                if freeu:
                    metadata |= {
                        'freeu_b1': freeu_b1, 'freeu_b2': freeu_b2, 'freeu_s1': freeu_s1, 'freeu_s2': freeu_s2
                    }
                if img2img_mode:
                    metadata |= {
                        'start_step': start_step, 'denoise': denoise, 'scale': img2img_scale, 'input_image': input_image_filename
                    }
                if revision_mode:
                    metadata |= {
                        'revision_strength_1': revision_strength_1, 'revision_strength_2': revision_strength_2,
                        'revision_strength_3': revision_strength_3, 'revision_strength_4': revision_strength_4,
                        'revision_images': revision_images_filenames
                    }
                if control_lora_canny:
                    metadata |= {
                        'canny_edge_low': canny_edge_low, 'canny_edge_high': canny_edge_high, 'canny_start': canny_start,
                        'canny_stop': canny_stop, 'canny_strength': canny_strength, 'canny_model': canny_model, 'canny_input': input_image_filename
                    }
                if control_lora_depth:
                    metadata |= {
                        'depth_start': depth_start, 'depth_stop': depth_stop, 'depth_strength': depth_strength, 'depth_model': depth_model, 'depth_input': input_image_filename
                    }
                metadata |= { 'software': fooocus_version.full_version }
    
                metadata_string = json.dumps(metadata, ensure_ascii=False)
                metadata_strings.append(metadata_string)
    
                for x in imgs:
                    d = [
                        ('Prompt', raw_prompt),
                        ('Negative Prompt', raw_negative_prompt),
                        ('Fooocus V2 (Prompt Expansion)', task['expansion']),
                        ('Styles', str(task['style_selections'])),
                        ('Real Prompt', task['positive']),
                        ('Real Negative Prompt', task['negative']),
                        ('Seed', task['task_seed']),
                        ('Resolution', get_resolution_string(width, height)),
                        ('Performance', (performance, steps, switch)),
                        ('Sampler & Scheduler', (sampler_name, scheduler)),
                        ('Sharpness', sharpness),
                        ('CFG & CLIP Skips', (cfg, base_clip_skip, refiner_clip_skip)),
                        ('Base Model', base_model_name),
                        ('Refiner Model', refiner_model_name),
                        ('FreeU', (freeu, freeu_b1, freeu_b2, freeu_s1, freeu_s2) if freeu else (freeu)),
                        ('Image-2-Image', (img2img_mode, start_step, denoise, img2img_scale, input_image_filename) if img2img_mode else (img2img_mode)),
                        ('Revision', (revision_mode, revision_strength_1, revision_strength_2, revision_strength_3,
                            revision_strength_4, revision_images_filenames) if revision_mode else (revision_mode)),
                        ('Prompt Strengths', (positive_prompt_strength, negative_prompt_strength)),
                        ('Canny', (control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop,
                            canny_strength, canny_model, input_image_filename) if control_lora_canny else (control_lora_canny)),
                        ('Depth', (control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, input_image_filename) if control_lora_depth else (control_lora_depth))
                    ]
                    for n, w in loras:
                        if n != 'None':
                            d.append((f'LoRA [{n}] weight', w))
                    d.append(('Software', fooocus_version.full_version))
                    d.append(('Execution Time', f'{execution_time:.2f} seconds'))
                    log(x, d, 3, metadata_string, save_metadata_json, save_metadata_image, keep_input_names, input_image_filename, output_format)

                results += imgs
            except comfy.model_management.InterruptProcessingException as e:
                print('User stopped')
                break

        outputs.append(['metadatas', metadata_strings])
        outputs.append(['results', results])

        pipeline.clear_all_caches() # cleanup after generation

        return

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
    pass


threading.Thread(target=worker, daemon=True).start()
