import gradio as gr
import random
import time
import shared
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import json
import modules.flags as flags
import modules.gradio_hijack as grh
import comfy.model_management as model_management

from modules.settings import default_settings
from modules.resolutions import get_resolution_string, resolutions
from modules.sdxl_styles import style_keys, fooocus_expansion, migrate_style_from_v1
from collections.abc import Mapping
from PIL import Image
from comfy.cli_args import args
from fastapi import FastAPI
from modules.ui_gradio_extensions import reload_javascript
from modules.util import get_current_log_path, get_previous_log_path
from modules.auth import auth_enabled, check_auth
from os.path import exists


GALLERY_ID_INPUT = 0
GALLERY_ID_REVISION = 1
GALLERY_ID_OUTPUT = 2


def generate_clicked(*args):
    execution_start_time = time.perf_counter()

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Initializing ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False), \
        gr.update(), \
        gr.update(value=None), \
        gr.update()

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(visible=False), \
                    gr.update(), \
                    gr.update(), \
                    gr.update()
            if flag == 'metadatas':
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=product), gr.update(selected=GALLERY_ID_OUTPUT)
            if flag == 'results':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True), \
                    gr.update(value=product), \
                    gr.update(), \
                    gr.update()
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def metadata_to_ctrls(metadata, ctrls):
    if not isinstance(metadata, Mapping):
        return ctrls

    if 'prompt' in metadata:
        ctrls[0] = metadata['prompt']
    if 'negative_prompt' in metadata:
        ctrls[1] = metadata['negative_prompt']
    if 'styles' in metadata:
        ctrls[2] = metadata['styles']
    elif 'style' in metadata:
        ctrls[2] = migrate_style_from_v1(metadata['style'])
    if 'performance' in metadata:
        ctrls[3] = metadata['performance']
    if 'width' in metadata and 'height' in metadata:
        ctrls[4] = get_resolution_string(metadata['width'], metadata['height'])
    elif 'resolution' in metadata:
        ctrls[4] = metadata['resolution']
    # image_number
    if 'seed' in metadata:
        ctrls[6] = metadata['seed']
        ctrls[60] = False
    if 'sharpness' in metadata:
        ctrls[7] = metadata['sharpness']
    if 'sampler_name' in metadata:
        ctrls[8] = metadata['sampler_name']
    elif 'sampler' in metadata:
        ctrls[8] = metadata['sampler']
    if 'scheduler' in metadata:
        ctrls[9] = metadata['scheduler']
    if 'steps' in metadata:
        ctrls[10] = metadata['steps']
        if ctrls[10] == constants.STEPS_SPEED:
            ctrls[3] = 'Speed'
        elif ctrls[10] == constants.STEPS_QUALITY:
            ctrls[3] = 'Quality'
        else:
            ctrls[3] = 'Custom'
    if 'switch' in metadata:
        ctrls[11] = round(metadata['switch'] / ctrls[10], 2)
        if ctrls[11] != round(constants.SWITCH_SPEED / constants.STEPS_SPEED, 2):
            ctrls[3] = 'Custom'
    if 'cfg' in metadata:
        ctrls[12] = metadata['cfg']
    if 'base_model' in metadata:
        ctrls[13] = metadata['base_model']
    elif 'base_model_name' in metadata:
        ctrls[13] = metadata['base_model_name']
    if 'refiner_model' in metadata:
        ctrls[14] = metadata['refiner_model']
    elif 'refiner_model_name' in metadata:
        ctrls[14] = metadata['refiner_model_name']
    if 'base_clip_skip' in metadata:
        ctrls[15] = metadata['base_clip_skip']
    if 'refiner_clip_skip' in metadata:
        ctrls[16] = metadata['refiner_clip_skip']
    if 'l1' in metadata:
        ctrls[17] = metadata['l1']
    if 'w1' in metadata:
        ctrls[18] = metadata['w1']
    if 'l2' in metadata:
        ctrls[19] = metadata['l2']
    if 'w2' in metadata:
        ctrls[20] = metadata['w2']
    if 'l3' in metadata:
        ctrls[21] = metadata['l3']
    if 'w3' in metadata:
        ctrls[22] = metadata['w3']
    if 'l4' in metadata:
        ctrls[23] = metadata['l4']
    if 'w4' in metadata:
        ctrls[24] = metadata['w4']
    if 'l5' in metadata:
        ctrls[25] = metadata['l5']
    if 'w5' in metadata:
        ctrls[26] = metadata['w5']
    # save_metadata_json
    # save_metadata_image
    if 'img2img' in metadata:
        ctrls[29] = metadata['img2img']
        if 'start_step' in metadata:
            if ctrls[3] == 'Speed':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_SPEED, 2)
            elif ctrls[3] == 'Quality':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_QUALITY, 2)
            else:
                ctrls[30] = round(metadata['start_step'] / ctrls[10], 2)
        if 'denoise' in metadata:
            ctrls[31] = metadata['denoise']
        if 'scale' in metadata:
            ctrls[32] = metadata['scale']
    if 'revision' in metadata:
        ctrls[33] = metadata['revision']
    if 'positive_prompt_strength' in metadata:
        ctrls[34] = metadata['positive_prompt_strength']
    elif 'zero_out_positive' in metadata:
        ctrls[34] = 0.0 if metadata['zero_out_positive'] else 1.0
    if 'negative_prompt_strength' in metadata:
        ctrls[35] = metadata['negative_prompt_strength']
    elif 'zero_out_negative' in metadata:
        ctrls[35] = 0.0 if metadata['zero_out_negative'] else 1.0
    if 'revision_strength_1' in metadata:
        ctrls[36] = metadata['revision_strength_1']
    if 'revision_strength_2' in metadata:
        ctrls[37] = metadata['revision_strength_2']
    if 'revision_strength_3' in metadata:
        ctrls[38] = metadata['revision_strength_3']
    if 'revision_strength_4' in metadata:
        ctrls[39] = metadata['revision_strength_4']
    # same_seed_for_all
    # output_format
    if 'control_lora_canny' in metadata:
        ctrls[42] = metadata['control_lora_canny']
    if 'canny_edge_low' in metadata:
        ctrls[43] = metadata['canny_edge_low']
    if 'canny_edge_high' in metadata:
        ctrls[44] = metadata['canny_edge_high']
    if 'canny_start' in metadata:
        ctrls[45] = metadata['canny_start']
    if 'canny_stop' in metadata:
        ctrls[46] = metadata['canny_stop']
    if 'canny_strength' in metadata:
        ctrls[47] = metadata['canny_strength']
    if 'canny_model' in metadata:
        ctrls[48] = metadata['canny_model']
    if 'control_lora_depth' in metadata:
        ctrls[49] = metadata['control_lora_depth']
    if 'depth_start' in metadata:
        ctrls[50] = metadata['depth_start']
    if 'depth_stop' in metadata:
        ctrls[51] = metadata['depth_stop']
    if 'depth_strength' in metadata:
        ctrls[52] = metadata['depth_strength']
    if 'depth_model' in metadata:
        ctrls[53] = metadata['depth_model']
    if 'prompt_expansion' in metadata:
        ctrls[54] = metadata['prompt_expansion']
    elif 'software' in metadata and metadata['software'].startswith('Fooocus 1.'):
        ctrls[54] = False
    if 'freeu' in metadata:
        ctrls[55] = metadata['freeu']
    if 'freeu_b1' in metadata:
        ctrls[56] = metadata['freeu_b1']
    if 'freeu_b2' in metadata:
        ctrls[57] = metadata['freeu_b2']
    if 'freeu_s1' in metadata:
        ctrls[58] = metadata['freeu_s1']
    if 'freeu_s2' in metadata:
        ctrls[59] = metadata['freeu_s2']
    # seed_random
    return ctrls    


def load_prompt_handler(_file, *args):
    ctrls=list(args)
    path = _file.name
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                print('load_prompt_handler, e: ' + str(e))
            finally:
                json_file.close()
    else:
        with open(path, 'rb') as image_file:
            image = Image.open(image_file)
            image_file.close()

            if path.endswith('.png') and 'Comment' in image.info:
                metadata_string = image.info['Comment']
            elif path.endswith('.jpg') and 'comment' in image.info:
                metadata_bytes = image.info['comment']
                metadata_string = metadata_bytes.decode('utf-8').split('\0')[0]
            else:
                metadata_string = None

            if metadata_string != None:
                try:
                    metadata = json.loads(metadata_string)
                    metadata_to_ctrls(metadata, ctrls)
                except Exception as e:
                    print('load_prompt_handler, e: ' + str(e))
    return ctrls


def load_last_prompt_handler(*args):
    ctrls = list(args)
    if exists(modules.path.last_prompt_path):
        with open(modules.path.last_prompt_path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                print('load_last_prompt_handler, e: ' + str(e))
            finally:
                json_file.close()
    return ctrls


def load_input_images_handler(files):
    return list(map(lambda x: x.name, files)), gr.update(selected=GALLERY_ID_INPUT), gr.update(value=len(files))


def load_revision_images_handler(files):
    return gr.update(value=True), list(map(lambda x: x.name, files[:4])), gr.update(selected=GALLERY_ID_REVISION)


def output_to_input_handler(gallery):
    if len(gallery) == 0:
        return [], gr.update()
    else:
        return list(map(lambda x: x['name'], gallery)), gr.update(selected=GALLERY_ID_INPUT)


def output_to_revision_handler(gallery):
    if len(gallery) == 0:
        return gr.update(value=False), [], gr.update()
    else:
        return gr.update(value=True), list(map(lambda x: x['name'], gallery[:4])), gr.update(selected=GALLERY_ID_REVISION)


settings = default_settings

app = FastAPI()
reload_javascript()

shared.gradio_root = gr.Blocks(title=fooocus_version.full_version, css=modules.html.css).queue()
with shared.gradio_root:
    with gr.Row():
        with gr.Column(scale=2):
            progress_window = grh.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False, elem_id='progress-bar', elem_classes='progress-bar')
            with gr.Column() as gallery_holder:
                with gr.Tabs(selected=GALLERY_ID_OUTPUT) as gallery_tabs:
                    with gr.Tab(label='Input', id=GALLERY_ID_INPUT):
                        input_gallery = gr.Gallery(label='Input', show_label=False, object_fit='contain', height=720, visible=True)
                    with gr.Tab(label='Revision', id=GALLERY_ID_REVISION):
                        revision_gallery = gr.Gallery(label='Revision', show_label=False, object_fit='contain', height=720, visible=True)
                    with gr.Tab(label='Output', id=GALLERY_ID_OUTPUT):
                        output_gallery = gr.Gallery(label='Output', show_label=False, object_fit='contain', height=720, visible=True)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder='What do you want to see.', container=False, autofocus=True, elem_classes='type_row', lines=1024, value=settings['prompt'])
                with gr.Column(scale=3, min_width=0):
                    with gr.Row():
                        img2img_mode = gr.Checkbox(label='Image-2-Image', value=settings['img2img_mode'], elem_classes='type_small_row')
                    with gr.Row():
                        generate_button = gr.Button(label='Generate', value='Generate', elem_classes='type_small_row', elem_id='generate_button', visible=True)
                        stop_button = gr.Button(label='Stop', value='Stop', elem_classes='type_small_row', elem_id='stop_button', visible=False)

                        def stop_clicked():
                            model_management.interrupt_current_processing()
                            return gr.update(interactive=False)

                        stop_button.click(fn=stop_clicked, outputs=stop_button, queue=False)

            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Enhance Image', value=False, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=settings['advanced_mode'], container=False, elem_classes='min_check')

            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Drag above image to here', source='upload', type='numpy')
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=flags.disabled)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='Inpaint or Outpaint (beta)') as inpaint_tab:
                        inpaint_input_image = grh.Image(label='Drag above image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF")
                        gr.HTML('Outpaint Expansion (<a href="https://github.com/lllyasviel/Fooocus/discussions/414">\U0001F4D4 Document</a>):')
                        outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint', show_label=False, container=False)
                        gr.HTML('* \"Inpaint or Outpaint\" is powered by the sampler \"DPMPP Fooocus Seamless 2M SDE Karras Inpaint Sampler\" (beta)')

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox, outputs=image_input_panel, queue=False,
                                        _js="(x) => {if(x){setTimeout(() => window.scrollTo({ top: window.scrollY + 500, behavior: 'smooth' }), 50);}else{setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 50);} return x}")

            current_tab = gr.Textbox(value='uov', visible=False)

            default_image = None

            def update_default_image(x):
                global default_image
                if isinstance(x, dict):
                    default_image = x['image']
                else:
                    default_image = x
                return

            def clear_default_image():
                global default_image
                default_image = None
                return

            uov_input_image.upload(update_default_image, inputs=uov_input_image, queue=False)
            inpaint_input_image.upload(update_default_image, inputs=inpaint_input_image, queue=False)

            uov_input_image.clear(clear_default_image, queue=False)
            inpaint_input_image.clear(clear_default_image, queue=False)

            uov_tab.select(lambda: ['uov', default_image], outputs=[current_tab, uov_input_image], queue=False)
            inpaint_tab.select(lambda: ['inpaint', default_image], outputs=[current_tab, inpaint_input_image], queue=False)

        with gr.Column(scale=1, visible=settings['advanced_mode']) as advanced_column:
            with gr.Tab(label='Settings'):
                performance = gr.Radio(label='Performance', choices=['Speed', 'Quality', 'Custom'], value=settings['performance'])
                with gr.Row(visible=settings['performance'] == 'Custom') as custom_row:
                    custom_steps = gr.Slider(label='Custom Steps', minimum=10, maximum=200, step=1, value=settings['custom_steps'])
                    custom_switch = gr.Slider(label='Custom Switch', minimum=0.2, maximum=1.0, step=0.01, value=settings['custom_switch'])
                resolution = gr.Dropdown(label='Resolution (width × height)', choices=list(resolutions.keys()), value=settings['resolution'], allow_custom_value=True)
                style_selections = gr.Dropdown(label='Image Style(s)', choices=style_keys, value=settings['styles'], multiselect=True, max_choices=8)
                with gr.Row():
                    prompt_expansion = gr.Checkbox(label=fooocus_expansion, value=settings['prompt_expansion'])
                    style_iterator = gr.Checkbox(label='Style Iterator', value=False)
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=256, step=1, value=settings['image_number'])
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="What you don't want to see.", value=settings['negative_prompt'])
                with gr.Row():
                   seed_random = gr.Checkbox(label='Random', value=settings['seed_random'])
                   same_seed_for_all = gr.Checkbox(label='Same seed for all images', value=settings['same_seed_for_all'])
                image_seed = gr.Textbox(label='Seed', value=settings['seed'], max_lines=1, visible=not settings['seed_random'])
                with gr.Row():
                    load_prompt_button = gr.UploadButton(label='Load Prompt', file_count='single', file_types=['.json', '.png', '.jpg'], elem_classes='type_small_row', min_width=0)
                    load_last_prompt_button = gr.Button(label='Load Last Prompt', value='Load Last Prompt', elem_classes='type_small_row', min_width=0)

                def get_current_links():
                    return '<a href="https://github.com/lllyasviel/Fooocus/discussions/117">&#128212; Fooocus Advanced</a>' \
                        + ' <a href="https://github.com/MoonRide303/Fooocus-MRE/wiki">&#128212; Fooocus-MRE Wiki</a>' \
                        + ' <a href="https://ko-fi.com/moonride" target="_blank">&#9749; Ko-fi</a> <br>' \
                        + f' <a href="/file={get_current_log_path()}" target="_blank">&#128212; Current Log</a>' \
                        + f' <a href="/file={get_previous_log_path()}" target="_blank">&#128212; Previous Log</a>'

                links = gr.HTML(value=get_current_links())

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    try:
                        seed_value = int(seed_string) 
                    except Exception as e:
                        seed_value = -1
                    if r or not isinstance(seed_value, int) or seed_value < constants.MIN_SEED or seed_value > constants.MAX_SEED:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        return seed_value

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed], queue=False)

                def performance_changed(value):
                    return gr.update(visible=value == 'Custom')

                performance.change(fn=performance_changed, inputs=[performance], outputs=[custom_row])

                def style_iterator_changed(_style_iterator, _style_selections):
                    if _style_iterator:
                        combinations_count = 1 + len(style_keys) - len(_style_selections) # original style selection + all remaining style combinations
                        return gr.update(interactive=False, value=combinations_count)
                    else:
                        return gr.update(interactive=True, value=settings['image_number'])

                def style_selections_changed(_style_iterator, _style_selections):
                    if _style_iterator:
                        combinations_count = 1 + len(style_keys) - len(_style_selections) # original style selection + all remaining style combinations
                        return gr.update(value=combinations_count)
                    else:
                        return gr.update()

                style_iterator.change(style_iterator_changed, inputs=[style_iterator, style_selections], outputs=[image_number])
                style_selections.change(style_selections_changed, inputs=[style_iterator, style_selections], outputs=[image_number])

            with gr.Tab(label='Image-2-Image'):
                revision_mode = gr.Checkbox(label='Revision (prompting with images)', value=settings['revision_mode'])
                revision_strength_1 = gr.Slider(label='Revision Strength for Image 1', minimum=-2, maximum=2, step=0.01,
                    value=settings['revision_strength_1'], visible=settings['revision_mode'])
                revision_strength_2 = gr.Slider(label='Revision Strength for Image 2', minimum=-2, maximum=2, step=0.01,
                    value=settings['revision_strength_2'], visible=settings['revision_mode'])
                revision_strength_3 = gr.Slider(label='Revision Strength for Image 3', minimum=-2, maximum=2, step=0.01,
                    value=settings['revision_strength_3'], visible=settings['revision_mode'])
                revision_strength_4 = gr.Slider(label='Revision Strength for Image 4', minimum=-2, maximum=2, step=0.01,
                    value=settings['revision_strength_4'], visible=settings['revision_mode'])

                def revision_changed(value):
                    return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True)

                revision_mode.change(fn=revision_changed, inputs=[revision_mode], outputs=[revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4])

                positive_prompt_strength = gr.Slider(label='Positive Prompt Strength', minimum=0, maximum=1, step=0.01, value=settings['positive_prompt_strength'])
                negative_prompt_strength = gr.Slider(label='Negative Prompt Strength', minimum=0, maximum=1, step=0.01, value=settings['negative_prompt_strength'])

                img2img_start_step = gr.Slider(label='Image-2-Image Start Step', minimum=0.0, maximum=0.8, step=0.01, value=settings['img2img_start_step'])
                img2img_denoise = gr.Slider(label='Image-2-Image Denoise', minimum=0.2, maximum=1.0, step=0.01, value=settings['img2img_denoise'])
                img2img_scale = gr.Slider(label='Image-2-Image Scale', minimum=1.0, maximum=2.0, step=0.25, value=settings['img2img_scale'],
                    info='For upscaling - use with low denoise values')

                keep_input_names = gr.Checkbox(label='Keep Input Names', value=settings['keep_input_names'], elem_classes='type_small_row')
                with gr.Row():
                    load_input_images_button = gr.UploadButton(label='Load Image(s) to Input', file_count='multiple', file_types=["image"], elem_classes='type_small_row', min_width=0)
                    load_revision_images_button = gr.UploadButton(label='Load Image(s) to Revision', file_count='multiple', file_types=["image"], elem_classes='type_small_row', min_width=0)
                with gr.Row():
                    output_to_input_button = gr.Button(label='Output to Input', value='Output to Input', elem_classes='type_small_row', min_width=0)
                    output_to_revision_button = gr.Button(label='Output to Revision', value='Output to Revision', elem_classes='type_small_row', min_width=0)

                load_input_images_button.upload(fn=load_input_images_handler, inputs=[load_input_images_button], outputs=[input_gallery, gallery_tabs, image_number])
                load_revision_images_button.upload(fn=load_revision_images_handler, inputs=[load_revision_images_button], outputs=[revision_mode, revision_gallery, gallery_tabs])
                output_to_input_button.click(output_to_input_handler, inputs=output_gallery, outputs=[input_gallery, gallery_tabs])
                output_to_revision_button.click(output_to_revision_handler, inputs=output_gallery, outputs=[revision_mode, revision_gallery, gallery_tabs])

                img2img_ctrls = [img2img_mode, img2img_start_step, img2img_denoise, img2img_scale, revision_mode, positive_prompt_strength, negative_prompt_strength,
                    revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4]

                def verify_revision(rev, gallery_in, gallery_rev, gallery_out):
                    if rev and len(gallery_rev) == 0:
                        if len(gallery_in) > 0:
                            gr.Info('Revision: imported input')
                            return gr.update(), list(map(lambda x: x['name'], gallery_in[:1]))
                        elif len(gallery_out) > 0:
                            gr.Info('Revision: imported output')
                            return gr.update(), list(map(lambda x: x['name'], gallery_out[:1]))
                        else:
                            gr.Warning('Revision: disabled (no images available)')
                            return gr.update(value=False), gr.update()
                    else:
                        return gr.update(), gr.update()

            with gr.Tab(label='CN'):
                control_lora_canny = gr.Checkbox(label='Control-LoRA: Canny', value=settings['control_lora_canny'])
                canny_edge_low = gr.Slider(label='Edge Detection Low', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['canny_edge_low'], visible=settings['control_lora_canny'])
                canny_edge_high = gr.Slider(label='Edge Detection High', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['canny_edge_high'], visible=settings['control_lora_canny'])
                canny_start = gr.Slider(label='Canny Start', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['canny_start'], visible=settings['control_lora_canny'])
                canny_stop = gr.Slider(label='Canny Stop', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['canny_stop'], visible=settings['control_lora_canny'])
                canny_strength = gr.Slider(label='Canny Strength', minimum=0.0, maximum=2.0, step=0.01,
                    value=settings['canny_strength'], visible=settings['control_lora_canny'])

                def canny_changed(value):
                    return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True), \
                        gr.update(visible=value == True), gr.update(visible=value == True)

                control_lora_canny.change(fn=canny_changed, inputs=[control_lora_canny], outputs=[canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength])

                control_lora_depth = gr.Checkbox(label='Control-LoRA: Depth', value=settings['control_lora_depth'])
                depth_start = gr.Slider(label='Depth Start', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['depth_start'], visible=settings['control_lora_depth'])
                depth_stop = gr.Slider(label='Depth Stop', minimum=0.0, maximum=1.0, step=0.01,
                    value=settings['depth_stop'], visible=settings['control_lora_depth'])
                depth_strength = gr.Slider(label='Depth Strength', minimum=0.0, maximum=2.0, step=0.01,
                    value=settings['depth_strength'], visible=settings['control_lora_depth'])

                def depth_changed(value):
                    return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True)

                control_lora_depth.change(fn=depth_changed, inputs=[control_lora_depth], outputs=[depth_start, depth_stop, depth_strength])

            with gr.Tab(label='Models'):
                with gr.Row():
                    base_model = gr.Dropdown(label='SDXL Base Model', choices=modules.path.model_filenames, value=settings['base_model'], show_label=True)
                    refiner_model = gr.Dropdown(label='SDXL Refiner', choices=['None'] + modules.path.model_filenames, value=settings['refiner_model'], show_label=True)
                with gr.Accordion(label='LoRAs', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'SDXL LoRA {i+1}', choices=['None'] + modules.path.lora_filenames, value=settings[f'lora_{i+1}_model'])
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=settings[f'lora_{i+1}_weight'])
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    canny_model = gr.Dropdown(label='Canny Model', choices=modules.path.canny_filenames, value=modules.path.default_controlnet_canny_name)
                    depth_model = gr.Dropdown(label='Depth Model', choices=modules.path.depth_filenames, value=modules.path.default_controlnet_depth_name)
                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')

                canny_ctrls = [control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength, canny_model]
                depth_ctrls = [control_lora_depth, depth_start, depth_stop, depth_strength, depth_model]

            with gr.Tab(label='Sampling'):
                cfg = gr.Slider(label='CFG', minimum=1.0, maximum=20.0, step=0.1, value=settings['cfg'])
                base_clip_skip = gr.Slider(label='Base CLIP Skip', minimum=-10, maximum=-1, step=1, value=settings['base_clip_skip'])
                refiner_clip_skip = gr.Slider(label='Refiner CLIP Skip', minimum=-10, maximum=-1, step=1, value=settings['refiner_clip_skip'])
                sampler_name = gr.Dropdown(label='Sampler', choices=['dpmpp_2m_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_3m_sde_gpu', 'dpmpp_3m_sde',
                    'dpmpp_sde_gpu', 'dpmpp_sde', 'dpmpp_2m', 'dpmpp_2s_ancestral', 'euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'ddpm'], value=settings['sampler'])
                scheduler = gr.Dropdown(label='Scheduler', choices=['karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform'], value=settings['scheduler'])
                sharpness = gr.Slider(label='Sampling Sharpness', minimum=0.0, maximum=30.0, step=0.01, value=settings['sharpness'])

                freeu_enabled = gr.Checkbox(label='FreeU', value=settings['freeu'])
                freeu_b1 = gr.Slider(label='Backbone Scaling Factor 1', minimum=0, maximum=2, step=0.01,
                    value=settings['freeu_b1'], visible=settings['freeu'])
                freeu_b2 = gr.Slider(label='Backbone Scaling Factor 2', minimum=0, maximum=2, step=0.01,
                    value=settings['freeu_b2'], visible=settings['freeu'])
                freeu_s1 = gr.Slider(label='Skip Scaling Factor 1', minimum=0, maximum=4, step=0.01,
                    value=settings['freeu_s1'], visible=settings['freeu'])
                freeu_s2 = gr.Slider(label='Skip Scaling Factor 2', minimum=0, maximum=4, step=0.01,
                    value=settings['freeu_s2'], visible=settings['freeu'])

                def model_refresh_clicked():
                    modules.path.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.path.model_filenames), gr.update(choices=['None'] + modules.path.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.path.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls, queue=False)

                def freeu_changed(value):
                    return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True), gr.update(visible=value == True)

                freeu_enabled.change(fn=freeu_changed, inputs=[freeu_enabled], outputs=[freeu_b1, freeu_b2, freeu_s1, freeu_s2])

                freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

            with gr.Tab(label='Misc'):
                output_format = gr.Radio(label='Output Format', choices=['png', 'jpg'], value=settings['output_format'])
                with gr.Row():
                    save_metadata_json = gr.Checkbox(label='Save Metadata in JSON', value=settings['save_metadata_json'])
                    save_metadata_image = gr.Checkbox(label='Save Metadata in Image', value=settings['save_metadata_image'])
                metadata_viewer = gr.JSON(label='Metadata')

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column, queue=False)

        def verify_enhance_image(enhance_image, img2img):
            if enhance_image and img2img:
                gr.Warning('Image-2-Image: disabled (Enhance Image priority)')
                return gr.update(value=False)
            else:
                return gr.update()

        def verify_input(img2img, canny, depth, gallery_in, gallery_rev, gallery_out):
            if (img2img or canny or depth) and len(gallery_in) == 0:
                if len(gallery_rev) > 0:
                    gr.Info('Image-2-Image / CL: imported revision as input')
                    return gr.update(), gr.update(), gr.update(), list(map(lambda x: x['name'], gallery_rev[:1]))
                elif len(gallery_out) > 0:
                    gr.Info('Image-2-Image / CL: imported output as input')
                    return gr.update(), gr.update(), gr.update(), list(map(lambda x: x['name'], gallery_out[:1]))
                else:
                    gr.Warning('Image-2-Image / CL: disabled (no images available)')
                    return gr.update(value=False), gr.update(value=False), gr.update(value=False), gr.update()
            else:
                return gr.update(), gr.update(), gr.update(), gr.update()

        ctrls = [
            prompt, negative_prompt, style_selections,
            performance, resolution, image_number, image_seed, sharpness, sampler_name, scheduler,
            custom_steps, custom_switch, cfg
        ]
        ctrls += [base_model, refiner_model, base_clip_skip, refiner_clip_skip] + lora_ctrls
        ctrls += [save_metadata_json, save_metadata_image] + img2img_ctrls + [same_seed_for_all, output_format]
        ctrls += canny_ctrls + depth_ctrls + [prompt_expansion] + freeu_ctrls
        load_prompt_button.upload(fn=load_prompt_handler, inputs=[load_prompt_button] + ctrls + [seed_random], outputs=ctrls + [seed_random])
        load_last_prompt_button.click(fn=load_last_prompt_handler, inputs=ctrls + [seed_random], outputs=ctrls + [seed_random])

        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image]
        ctrls += [style_iterator]
        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False), []), outputs=[stop_button, generate_button, output_gallery]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=verify_enhance_image, inputs=[input_image_checkbox, img2img_mode], outputs=[img2img_mode]) \
            .then(fn=verify_input, inputs=[img2img_mode, control_lora_canny, control_lora_depth, input_gallery, revision_gallery, output_gallery],
                outputs=[img2img_mode, control_lora_canny, control_lora_depth, input_gallery]) \
            .then(fn=verify_revision, inputs=[revision_mode, input_gallery, revision_gallery, output_gallery], outputs=[revision_mode, revision_gallery]) \
            .then(fn=generate_clicked, inputs=ctrls + [input_gallery, revision_gallery, keep_input_names],
                outputs=[progress_html, progress_window, gallery_holder, output_gallery, metadata_viewer, gallery_tabs]) \
            .then(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[generate_button, stop_button]) \
            .then(fn=get_current_links, inputs=None, outputs=links) \
            .then(fn=None, _js='playNotification')

        notification_file = 'notification.ogg' if exists('notification.ogg') else 'notification.mp3' if exists('notification.mp3') else None
        if notification_file != None:
            gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)


app = gr.mount_gradio_app(app, shared.gradio_root, '/')
shared.gradio_root.launch(inbrowser=True, server_name=args.listen, server_port=args.port, share=args.share,
    auth=check_auth if args.share and auth_enabled else None, allowed_paths=[modules.path.temp_outputs_path])
