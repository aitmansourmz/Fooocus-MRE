### 2.0.78.5 MRE

* Added Style Iterator.
* Removed meta tensor usage.
* Fixed loading TAESD decoder for SD from custom path.

### 2.0.78.4 MRE

* Added Load Last Prompt button (contribution from sngazm).
* Fixed hangup in Upscale (Fast 2x).
* Fixed problems with turning off FreeU in some scenarios.
* Fixed loading prompts from JPG files processed by external apps.
* Fixed fast upscale always saving as PNG.

### 2.0.78.3 MRE

* Added limited support for non-SDXL models (no refiner, Control-LoRAs, Revision, inpainting, outpainting).

### 2.0.78.2 MRE

* Added support for FreeU.
* Updated Comfy.

### 2.0.78.1 MRE

* Fixed reading paths from paths.json (broken in 2.0.73 MRE).
* Fixed error related to playing audio notification.
* Fixed error related to loading prompt with Enhance Image mode active.
* Disabling Image-2-Image when Enhance Image is active.

### 2.0.76 MRE

* Added information about total execution time.
* Enforced 'dpmpp_fooocus_2m_sde_inpaint_seamless' sampler for inpainting workflow.
* Updated Comfy and patched Comfy functions.

### 2.0.73 MRE

* Renamed Input Image from vanilla to Enhance Image (to avoid confusion with Input tab)

### 2.0.19 MRE

* Added support for wildcards (ported from RuinedFooocus, adjusted to Fooocus V2).
* Added support for ddpm sampler.
* Restored saving information about real prompt in metadata and log file (adjusted to Fooocus V2).
* Fixed links to log files not working with customized outputs path.
* Disabled Fooocus Virtual Memory from vanilla (not compatible with current Comfy).
* Updated Comfy.

### 2.0.18 MRE

* Added support for authentication in --share mode (via auth.json).
* Added Image-2-Image Scale slider.
* Displaying Revision and Control-LoRAs controls only when needed.

### 2.0.14 MRE

* Added support for loading models from subfolders (ported from RuinedFooocus).
* Updated Comfy.

### 2.0.12 MRE

* Added support for higher resolutions in Image-2-Image mode (can be used for upscaling).

### 2.0.3 MRE

* Updated Comfy (CLIP Vision optimizations).

### 2.0.0 MRE

* Changed Prompt Expansion (aka Fooocus V2) to be enabled by default.
* Moved links to Settings tab.

### 1.0.67 MRE

* Updated Comfy (fixes CLIP issue).

### 1.0.61 MRE

* Restored allowed random seed range (entropy reduction applied only to transformers / numpy related calls).

### 1.0.51 MRE

* Added support for adjusting text prompt strengths (useful in Revision mode).
* Reduced allowed random seed range to match limits in Prompt Expansion and transformers (trainer_utils.py).
* Updated Comfy.

### 1.0.50 MRE

* Renamed Raw Mode (enabled by default in vanilla) to Prompt Expansion (disabled by default in MRE).

### 1.0.45.1 MRE

* Added support for reading styles from JSON files.
* Added support for playing audio when generation is finished (ported from SD web UI).
* Fixed joining negative prompts.
* Increased Control-LoRAs strength range.
* Allowed passing parameters to Comfy.
* Added links in Misc tab
* Updated Comfy.

### 1.0.45 MRE

* Added support for custom resolutions and custom resolutions list.
* Updated Comfy.
* Added MRE changelog.

### 1.0.43 MRE

* Added support for Control-LoRA: Depth.
* Added Canny and Depth model selection.
* Added ability to stop image generation.
* Added support for generate forever mode (ported from SD web UI).
* Added information about prompt execution time.
* Changed default Control-LoRA rank to 128 (allows using both on 8 GB VRAM GPUs).
* Fixed problems with random seed precision.
* Increased maximum number of generated images to 128.
* Moved Revision in front of Control-LoRAs.
* Updated Comfy.
* Updated Gradio.
* Updated Colab notebook.

### 1.0.41 MRE

* Added support for Revision (thx to comfyanonymous for helping figuring out issues related to lllyasviel changes).
* Added support for Control-LoRA: Canny.
* Added support for Embeddings.
* Added support for JPEGs (with metadata saving & loading).
* Added support for generating multiple images using same seed (useful in Image-2-Image mode).
* Added support for keeping input files names.
* Added information about aspect ratio to resolution list.
* Added dpmpp_2m sampler.
* Automatically import images into Image-2-Image and Revision modes (if available).
* Automatically set batch size to number of loaded input images.
* Loading only base model on start.
* Updated Comfy.
* Updated Colab notebook.

### 1.0.40 MRE

* Added support for Image-2-Image mode.
* Added support for changing paths (via paths.json).
* Added Custom performance mode.
* Fixed prompt loading.
* Implemented cycling over input images in Image-2-Image mode.
* Made resolution and style selection more compact (thx to runew0lf for hints).
* Simplified initial view.
* Updated Comfy.
* Improvemed VRAM management.
* Corrected Colab notebook.

### 1.0.39 MRE

* Updated Comfy (+DISABLE_SMART_MEMORY).

### 1.0.36 MRE

* Fixed problem with disabled refiner (by youcheng).
* Layout corrections.
* Updated documentation.
* Renamed fork to "Fooocus-MRE".

### 1.0.35 MRE

* Added support for viewing metadata in the UI.
* Added support for loading basic UI settings from file (settings.json).
* Allowed saving metadata in both JSON and PNG file.
* Fixed problem with handling text files encoding (for non-ASCII prompts).

### 1.0.32 MRE

* Added support for loading prompt parameters from JSON and PNG files.
* Added support for changing scheduler (karras, exponential, simple, ddim_uniform).
* Enhanced list of available samplers (added dpmpp_sde_gpu, dpmpp_sde, dpmpp_2s_ancestral, euler, euler_ancestral, heun, dpm_2, dpm_2_ancestral).
* Updated Comfy.

### 1.0.31 MRE

* Added support for dpmpp_3m_sde_gpu, dpmpp_3m_sde, and dpmpp_2m_sde samplers.
* Added support for saving prompt information in PNG metadata or JSON file.
* Added support for changing steps, switch step, and CFG.
* Added support for changing CLIP Skip values.
* Changed resolution list to official SDXL resolutions.
* Enhanced RNG seed range.
