import os
import requests
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from PIL import Image
from io import BytesIO


class Predictor:
    def setup(self) -> None:
        model_path = './llava-v1.5-13b'
        torch_dtype = torch.float16
        disable_torch_init()
        torch.backends.cudnn.benchmark = True

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            torch_dtype=torch_dtype,
            model_name="llava-v1.5-13b",
            load_in_8bit=False,
            load_in_4bit=False
        )

        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)

    def predict(
            self,
            image: str,
            prompt: str,
            top_p: float = 0.9,
            num_beams: float = 1,
            temperature: float = 0.1,
            max_tokens: int = 128,
    ) -> str:
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_data = load_image(image)
        image_tensor = self.image_processor.preprocess(
            image_data, return_tensors='pt'
        )['pixel_values'].half().to(self.device)

        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_tokens,
                use_cache=True
            )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output_text


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image