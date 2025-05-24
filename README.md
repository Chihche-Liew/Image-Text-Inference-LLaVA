# Image-Text-Inference-LLaVA
This repository contains a Python script for running multimodal inference using the LLaVA v1.5 model. It accepts an image and a prompt, and returns a generated response using vision-language capabilities.

## Features
- Supports loading local or remote images.

- Uses `llava`'s tokenizer and image processor.

- Outputs model-generated text based on image+prompt input.

- Runs on GPU with `float16` precision for efficiency.

## Tested Environment
- CPU: Intel(R) Xeon(R) Gold 6248R @ 3.00GHz

- RAM: 753 GB

- GPU: NVIDIA Tesla T4

- Python: 3.10

- CUDA: 12.1

- Conda: Minimal required packages listed below

## Dependencies
Only essential Python dependencies required for this script:
```
pip install torch==2.1.2 torchvision==0.16.2
pip install llava==1.2.2.post1 transformers==4.37.2
pip install pillow requests
```
If you're using `conda`, create a minimal environment with:
```
conda create -n llava python=3.10
conda activate llava
pip install torch torchvision llava transformers pillow requests
```

## Usage
### 1. Place your LLaVA model
Download the LLaVA v1.5 -xB weights and place them at ./llava-v1.5-xb.
### 2. Run the script
```{python}
predictor = Predictor()
predictor.setup()
response = predictor.predict(image='your_image.jpg', prompt='What is happening in this image?')
print(response)
```
You can also use remote image URLs.

## Notes
- This script is tailored for `LLaVA v1.5`. It may require minor changes for `LLaVA-2` compatibility (e.g., conv_templates, tokenizer updates).

- The model runs in `float16` on CUDA-enabled GPUs. If you're using a different setup, adjust torch_dtype accordingly.

- The model uses the default conversation template `llava_v1`.

