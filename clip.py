from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

