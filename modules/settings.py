import json
import modules.path

from modules.resolutions import get_resolution_string
from os.path import exists


def load_settings():
    settings = {}
    settings['advanced_mode'] = False
    settings['image_number'] = 2
    settings['save_metadata_json'] = False
    settings['save_metadata_image'] = False
    settings['output_format'] = 'png'
    settings['seed_random'] = True
    settings['same_seed_for_all'] = False
    settings['seed'] = 0
    settings['styles'] = ['Default (Slightly Cinematic)']
    settings['prompt_expansion'] = True
    settings['prompt'] = ''
    settings['negative_prompt'] = ''
    settings['performance'] = 'Speed'
    settings['custom_steps'] = 24
    settings['custom_switch'] = 0.75
    settings['img2img_mode'] = False
    settings['img2img_start_step'] = 0.06
    settings['img2img_denoise'] = 0.94
    settings['img2img_scale'] = 1.0
    settings['control_lora_canny'] = False
    settings['canny_edge_low'] = 0.2
    settings['canny_edge_high'] = 0.8
    settings['canny_start'] = 0.0
    settings['canny_stop'] = 0.4
    settings['canny_strength'] = 0.8
    settings['canny_model'] = modules.path.default_controlnet_canny_name
    settings['control_lora_depth'] = False
    settings['depth_start'] = 0.0
    settings['depth_stop'] = 0.4
    settings['depth_strength'] = 0.8
    settings['depth_model'] = modules.path.default_controlnet_depth_name
    settings['keep_input_names'] = False
    settings['revision_mode'] = False
    settings['positive_prompt_strength'] = 1.0
    settings['negative_prompt_strength'] = 1.0
    settings['revision_strength_1'] = 1.0
    settings['revision_strength_2'] = 1.0
    settings['revision_strength_3'] = 1.0
    settings['revision_strength_4'] = 1.0
    settings['resolution'] = get_resolution_string(1152, 896)
    settings['sampler'] = 'dpmpp_2m_sde_gpu'
    settings['scheduler'] = 'karras'
    settings['cfg'] = 7.0
    settings['base_clip_skip'] = -2
    settings['refiner_clip_skip'] = -2
    settings['sharpness'] = 2.0
    settings['base_model'] = modules.path.default_base_model_name
    settings['refiner_model'] = modules.path.default_refiner_model_name
    settings['lora_1_model'] = modules.path.default_lora_name
    settings['lora_1_weight'] = modules.path.default_lora_weight
    settings['lora_2_model'] = 'None'
    settings['lora_2_weight'] = modules.path.default_lora_weight
    settings['lora_3_model'] = 'None'
    settings['lora_3_weight'] = modules.path.default_lora_weight
    settings['lora_4_model'] = 'None'
    settings['lora_4_weight'] = modules.path.default_lora_weight
    settings['lora_5_model'] = 'None'
    settings['lora_5_weight'] = modules.path.default_lora_weight
    settings['freeu'] = False
    settings['freeu_b1'] = 1.01
    settings['freeu_b2'] = 1.02
    settings['freeu_s1'] = 0.99
    settings['freeu_s2'] = 0.95

    if exists('settings.json'):
        with open('settings.json') as settings_file:
            try:
                settings_obj = json.load(settings_file)
                for k in settings.keys():
                    if k in settings_obj:
                        settings[k] = settings_obj[k]
            except Exception as e:
                print('load_settings, e: ' + str(e))
            finally:
                settings_file.close()

    return settings


default_settings = load_settings()
