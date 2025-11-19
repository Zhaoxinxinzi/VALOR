from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import numpy as np
import torch

class ImageSafetyChecker:
    def __init__(self, model_path="file_path"):
        print(f"Loading Stable Diffusion Safety Checker from {model_path}")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.checker = StableDiffusionSafetyChecker.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checker.to(self.device)

    def is_image_safe(self, image_path: str) -> dict:
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            inputs = self.extractor(images=image_np, return_tensors="pt").to(self.device)

            with torch.no_grad():
                _, has_nsfw_concepts = self.checker(
                    images=[image_np],
                    clip_input=inputs.pixel_values.to(self.device)
                )

            return {
                "image_path": image_path,
                "is_safe": not has_nsfw_concepts[0],
                "nsfw_flag": bool(has_nsfw_concepts[0])
            }

        except Exception as e:
            return {
                "image_path": image_path,
                "is_safe": False,
                "error": str(e)
            }
