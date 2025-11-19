import os
import torch
from typing import List
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    PixArtAlphaPipeline
)

try:
    from diffusers import StableDiffusion3Pipeline
    SD3_AVAILABLE = True
except ImportError:
    print("StableDiffusion3Pipeline not available, please update diffusers: pip install --upgrade diffusers")
    SD3_AVAILABLE = False

torch.backends.cuda.matmul.allow_tf32 = True

SAFETY_GUIDANCE_SUFFIX = "in a detailed, safe-for-work artistic painting style, such as watercolor or digital illustration"

class DiffusionImageGenerator:
    def __init__(self, model_type: str = "sdxl", model_path: str = None, device_id: int = None):
        self.model_type = model_type.lower()
        self.model_path = model_path or self.get_default_path(self.model_type)
        
        self.device_id = device_id
        self.device = self._setup_device()
        
        if self.model_type in ["sd3", "sd35"] and not SD3_AVAILABLE:
            raise ValueError(f"SD3 model requires updated diffusers library: pip install --upgrade diffusers")
        
        self.pipe = self.load_pipeline()

    def _setup_device(self) -> torch.device:
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device("cpu")
        
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPUs")
        
        if self.device_id is not None:
            if self.device_id >= gpu_count:
                print(f"Specified GPU {self.device_id} does not exist, only {gpu_count} GPUs available")
                print(f"   Using GPU 0 instead")
                device = torch.device(f"cuda:0")
            else:
                device = torch.device(f"cuda:{self.device_id}")
                print(f"Using specified GPU {self.device_id}")
        else:
            device = torch.device("cuda:0")
            print(f"Auto-selecting GPU 0")
        
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device.index)
            gpu_memory = torch.cuda.get_device_properties(device.index).total_memory / 1024**3
            print(f"Using GPU: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
            
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        
        return device

    def get_default_path(self, model_type: str) -> str:
        paths = {
            "sd14": "file_path/stable-diffusion-v1-4",
            "sd21": "file_path/stable-diffusion-v2-1", 
            "sdxl": "file_path/stable-diffusion-xl-base1.0",
            "sd3": "file_path/stable-diffusion-3.5-large",
            "sd35": "file_path/stable-diffusion-3.5-large",
            "sd3-medium": "file_path/stable-diffusion-3-medium",
            "pixart": "file_path/PixArt-alpha"
        }
        return paths[model_type]

    def load_pipeline(self):
        print(f"Loading model: {self.model_type} from {self.model_path}")
        print(f"Target device: {self.device}")
        
        if self.model_type in ["sd3", "sd35", "sd3-medium"]:
            if not SD3_AVAILABLE:
                raise ValueError("SD3 Pipeline not available, please update diffusers library")
            
            print(f"Using StableDiffusion3Pipeline to load SD3 model...")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
        elif self.model_type in ["sdxl"]:
            print(f"Using StableDiffusionXLPipeline to load SDXL model...")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
        elif self.model_type in ["sd14", "sd21"]:
            print(f"Using StableDiffusionPipeline to load SD1.x/2.x model...")
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
        elif self.model_type in ["pixart"]:
            print(f"Using PixArtAlphaPipeline to load pixart model...")
            pipe = PixArtAlphaPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        pipe = pipe.to(self.device)
        print(f"Model loaded to {self.device}")
        
        if self.model_type in ["sdxl", "sd3", "sd35"]:
            pipe.enable_attention_slicing()
            print(f"Enabled attention slicing optimization")
        
        if self.model_type in ["sd3", "sd35"]:
            pipe.enable_model_cpu_offload()
            print(f"Enabled CPU offload optimization")
        
        return pipe

    def truncate_prompt(self, prompt: str) -> str:
        if self.model_type in ["sd3", "sd35", "sd3-medium"]:
            max_length = 256
        else:
            max_length = 77
        
        words = prompt.split()
        if len(words) > max_length:
            truncated = " ".join(words[:max_length])
            print(f"Prompt truncated from {len(words)} to {max_length} words")
            return truncated
        return prompt

    def generate_images(self, prompts: List[str], output_dir: str = "outputs/sd14/", batch_size: int = 10) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {len(prompts)} images in batches of {batch_size} with {self.model_type.upper()}...")
        print(f"Using device: {self.device}")

        output_paths = []
        existing_imgs = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
        base_index = len(existing_imgs)
        
        if self.model_type in ["sd3", "sd35"]:
            num_inference_steps = 20
            guidance_scale = 7.0
        elif self.model_type == "sdxl":
            num_inference_steps = 25
            guidance_scale = 7.5
        else:
            num_inference_steps = 30
            guidance_scale = 7.5
        
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            truncated_prompts = [self.truncate_prompt(p) for p in batch_prompts]

            try:
                print(f"Generating batch {start//batch_size + 1}: {len(batch_prompts)} images...")
                
                with torch.cuda.device(self.device):
                    if self.model_type in ["sd3", "sd35"]:
                        images = self.pipe(
                            prompt=truncated_prompts, 
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=1024,
                            width=1024
                        ).images
                    else:
                        images = self.pipe(
                            prompt=truncated_prompts, 
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale
                        ).images
                
                for i, image in enumerate(images):
                    index = base_index + start + i
                    path = os.path.join(output_dir, f"{self.model_type}_output_{index}.png")
                    image.save(path)
                    output_paths.append(path)
                    print(f"Saved: {path}")
                    
            except Exception as e:
                print(f"Failed to generate images for batch {start}-{end}: {e}")
                print(f"If SD3 error, try:")
                print(f"   1. Update diffusers: pip install --upgrade diffusers")
                print(f"   2. Check model file integrity")
                print(f"   3. Try other models: --image-model sdxl")
                print(f"   4. Check GPU memory")
            finally:
                if "images" in locals():
                    del images
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return output_paths
    
    def regenerate_safe_image(self, prompt: str, output_path: str) -> str:
        safe_prompt = prompt + " " + SAFETY_GUIDANCE_SUFFIX
        print(f"Regenerating with safety prompt: {safe_prompt}")
        print(f"Using device: {self.device}")

        with torch.cuda.device(self.device):
            if self.model_type in ["sd3", "sd35"]:
                image = self.pipe(
                    prompt=safe_prompt, 
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    height=1024,
                    width=1024
                ).images[0]
            else:
                image = self.pipe(prompt=safe_prompt, num_inference_steps=20).images[0]
                
        safe_path = output_path
        image.save(safe_path)
        print(f"Saved corrected image to: {safe_path}")
        return safe_path

    def get_device_info(self) -> dict:
        info = {
            "device": str(self.device),
            "device_id": self.device_id,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available() and self.device.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(self.device.index),
                "gpu_memory_total": torch.cuda.get_device_properties(self.device.index).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device.index) / 1024**3,
                "gpu_memory_cached": torch.cuda.memory_reserved(self.device.index) / 1024**3
            })
        
        return info