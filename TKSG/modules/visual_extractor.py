import torch
import torch.nn as nn
import torchvision.models as models
from CLIP import clip

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.model_frame = args.model_frame
        self.clip = args.clip

        if self.model_frame == 'ResNet':
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        if self.model_frame == 'clip_feat':
            self.avg_fnt_1 = torch.nn.AvgPool1d(kernel_size=196, stride=196)

        elif self.model_frame == 'finetune_clip':

            self.avg_fnt_1 = torch.nn.AvgPool1d(kernel_size=196, stride=196)
            self.clip_model, self.preprocess = clip.load(self.clip, device=args.device)
            self.clip_model.visual.proj = nn.Parameter(torch.randn(768, args.concept_dim))  

            self.clip_model.requires_grad_(False)  

            # resblocks = self.clip_model.visual.transformer.resblocks
            # for block in resblocks[-2:]:
            #     for param in block.parameters():
            #         param.requires_grad = True

            self.clip_model.visual.proj.requires_grad_(True)  

        elif self.model_frame == 'ViT':
            self.vit = models.vit_b_32(pretrained=True)
            self.vit.heads = nn.Identity()
            self.att_feat_size = 768
            self.output_proj = nn.Linear(self.att_feat_size, 2048)  # Project the features if needed
            nn.init.xavier_uniform_(self.output_proj.weight) 
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
        else:
            if self.model_frame == 'swin_transformer_base': # Load the pretrained Swin-transformer model from torchvision
                self.swin = models.swin_b(pretrained=True)
                self.att_feat_size = 1024
            elif self.model_frame == 'swin_transformer_small':
                self.swin = models.swin_s(pretrained=True)
                self.att_feat_size = 768
            else: # 'swin_transformer_tiny'
                self.swin = models.swin_t(pretrained=True)
                self.att_feat_size = 768
            self.swin.head = nn.Identity()  # Remove the final classification head
            self.output_proj = nn.Linear(self.att_feat_size, 2048)  # Project the features if needed
            nn.init.xavier_uniform_(self.output_proj.weight) 
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(self, images):
        #################################### ResNet ###################################
        if self.model_frame == 'ResNet':
            patch_feats = self.model(images)
            batch_size, feat_size, _, _ = patch_feats.shape

        ####################################### load CLIP data###################################
        if self.model_frame == 'clip_feat':
            patch_feats = images

        ################################## finetune CLIP ##################################
        if self.model_frame == 'finetune_clip':
            patch_feats = self.clip_model.encode_image(images)  # (8,196,2048)

        ################################## ViT ##################################
        elif self.model_frame == 'ViT':
            x = self.vit._process_input(images)  
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.vit.class_token.expand(n, -1, -1) # CLS head
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.vit.encoder(x)
            patch_feats = x[:, 1:, :]  
            patch_feats = self.output_proj(patch_feats)

        ################################## Swin transformer ##################################
        elif self.model_frame in ['swin_transformer_base', 'swin_transformer_small', 'swin_transformer_tiny']:
            features = self.swin.features(images)
            features = self.swin.norm(features)

            features = features.permute(0, 3, 1, 2)  # Now shape: (B, dim, 7, 7)
            # Flatten the features into (B, patch, dim), where patch = 7*7 and dim
            patch_feats = features.flatten(2).transpose(1, 2)  # Shape: (B, 49, dim)
            # Optionally, project the features to a different dimension if needed
            patch_feats = self.output_proj(patch_feats)

        # avg_feats = self.avg_fnt_1(patch_feats.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        if self.model_frame == 'ResNet':
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        else:
            avg_feats = patch_feats.mean(dim=1)

        return patch_feats, avg_feats
