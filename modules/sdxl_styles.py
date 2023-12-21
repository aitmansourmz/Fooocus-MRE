import os
import json
import re
import random

from os.path import exists
from modules.path import styles_path, wildcards_path, get_files_from_folder
from modules.util import join_prompts


fooocus_expansion = "Prompt Expansion (Fooocus V2)"


base_styles = [
    {
        "name": "Default (Slightly Cinematic)",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    }
]


default_styles_files = ['sdxl_styles_sai.json', 'sdxl_styles_twri.json', 'sdxl_styles_diva.json', 'sdxl_styles_mre.json']


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


def migrate_style_from_v1(style):
    if style == 'cinematic-default':
        return ['Default (Slightly Cinematic)']
    elif style == 'None':
        return []
    else:
        return [normalize_key(style)]


def styles_list_to_styles_dict(styles_list=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    if isinstance(styles_list, list) and len(styles_list) > 0:
        for entry in styles_list:
            name, prompt, negative_prompt = normalize_key(entry['name']), entry['prompt'], entry['negative_prompt']
            if name not in styles_dict:
                styles_dict |=  {name: (prompt, negative_prompt)}
    return styles_dict


def load_styles(filename=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    full_path = os.path.join(styles_path, filename) if filename != None else None
    if full_path != None and exists(full_path):
        with open(full_path, encoding='utf-8') as styles_file:
            try:
                styles_obj = json.load(styles_file)
                styles_list_to_styles_dict(styles_obj, styles_dict)
            except Exception as e:
                print('load_styles, e: ' + str(e))
            finally:
                styles_file.close()
    return styles_dict


styles = styles_list_to_styles_dict(base_styles)
for styles_file in default_styles_files:
    styles = load_styles(styles_file, styles)


all_styles_files = get_files_from_folder(styles_path, ['.json'])
for styles_file in all_styles_files:
    if styles_file not in default_styles_files:
        styles = load_styles(styles_file, styles)


style_keys = list(styles.keys())


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive), n



def apply_wildcards(wildcard_text, seed=None, directory=wildcards_path):
    placeholders = re.findall(r'__(\w+)__', wildcard_text)
    for placeholder in placeholders:
        try:
            with open(os.path.join(directory, f'{placeholder}.txt')) as f:
                words = f.read().splitlines()
                f.close()
                rng = random.Random(seed)
                wildcard_text = re.sub(rf'__{placeholder}__', rng.choice(words), wildcard_text)
        except IOError:
            print(f'Error: could not open wildcard file {placeholder}.txt, using as normal word.')
            wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
    return wildcard_text
