from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel
import torch
from torch import nn
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

class CLIPVisualPrompt:
    def __init__(self, clip_path, feature_dim=768):
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_path, subfolder="vision_encoder")
        self.text_model = CLIPTextModel.from_pretrained(clip_path, subfolder="text_encoder")
        self.clip_mlp = IM2TEXT(feature_dim, 768, feature_dim)
        
    def train_vtoken(self, input_ids, image_features):
        
        B, L = input_ids.shape
        text_hidden_states = self.text_model(input_ids).last_hidden_state
        image_features = self.vision_model(image_features).image_embeds.unsqueeze(1)
        image_features = self.clip_mlp(image_features)
        
        hidden_states = torch.cat((text_hidden_states[:, :-2, :], image_features, text_hidden_states[:, -1, :].unsqueeze(1)), dim=1)
        
        return hidden_states
    
    def uncondition_train_vtoken(self, input_ids, image_features):
        
        B, L = input_ids.shape
        text_hidden_states = self.text_model(input_ids).last_hidden_state
        image_features = self.vision_model(torch.zeros_like(image_features)).image_embeds.unsqueeze(1)
        image_features = self.clip_mlp(image_features)
        
        hidden_states = torch.cat((text_hidden_states[:, 0, :].unsqueeze(1), image_features, text_hidden_states[:, 1:-1, :]), dim=1)
        
        return hidden_states
    
    @torch.inference_mode
    def inference_vtoken(self, input_ids, input_ids_uncond, image, text_encoder):
        
        B, L = input_ids.shape
        text_hidden_states = text_encoder(input_ids).last_hidden_state
        uncond_text_hidden_states = text_encoder(input_ids_uncond).last_hidden_state
        
        image_features = self.vision_model(image).image_embeds.unsqueeze(1)
        image_features = self.clip_mlp(image_features)
        
        uncond_image_features = self.vision_model(torch.zeros_like(image)).image_embeds.unsqueeze(1)
        uncond_image_features = self.clip_mlp(uncond_image_features)
        
        text_hidden_states = torch.cat((text_hidden_states[:, :-2, :], image_features, text_hidden_states[:, -1, :].unsqueeze(1)), dim=1)
        uncond_text_hidden_states = torch.cat((uncond_text_hidden_states[:, 0, :].unsqueeze(1), uncond_image_features, uncond_text_hidden_states[:, 1:-1, :]), dim=1)
        
        return text_hidden_states, uncond_text_hidden_states
        
    def load_mlp_weight(self, file_dir):
        clip_mlp_weight = torch.load(file_dir)
        self.clip_mlp.load_state_dict(clip_mlp_weight)

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=1024, middle_dim=1024, output_dim=1024):
        super().__init__()
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor):

        return x + self.fc_out(self.norm(x))
    