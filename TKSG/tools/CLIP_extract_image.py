# 参考：https://github.com/mlfoundations/open_clip
import torch
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import clip

avgpool = True 
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
model, preprocess = clip.load("ViT-B/16", device=device)

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune

json_file_path = f'/dataset/{dataset_name}/annotation.json'
image_dir = f'/dataset/{dataset_name}/images/'
if avgpool:
    image_feat_path = f'/dataset/{dataset_name}/CLIP_image(1×512)-vit-b-16.npz'
else:
    image_feat_path = f'/dataset/{dataset_name}/CLIP_image(196×512)-vit-b-16.npz'
print('annotation path:', json_file_path)
print('iamge path:', image_dir)
print('image feats save path:', image_feat_path)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)

image_path = []
for split in ['train','val','test']:
    data = json_data[split]
    for i in data:
        image_path.append(i['image_path'])

total_feat = {}
for data in tqdm(image_path):
    with torch.no_grad():
        if dataset_name == 'iu_xray-finetune':
            images0 = preprocess(Image.open(image_dir + data[0])).unsqueeze(0).to(device)
            images1 = preprocess(Image.open(image_dir + data[1])).unsqueeze(0).to(device)
            if avgpool is False:
                image_features0 = model.encode_image(images0).squeeze().cpu().numpy()  # (196,512)
                total_feat.update({data[0]: image_features0})
                image_features1 = model.encode_image(images1).squeeze().cpu().numpy()
                total_feat.update({data[1]: image_features1})
            if avgpool is True:
                image_features0 = model.encode_image(images0).squeeze().mean(dim=0, keepdim=True).cpu().numpy()  # (1,512)
                total_feat.update({data[0]: image_features0})
                image_features1 = model.encode_image(images1).squeeze().mean(dim=0, keepdim=True).cpu().numpy()  # (1,512)
                total_feat.update({data[1]: image_features1})
        if dataset_name == 'mimic_cxr-512':
            images = preprocess(Image.open(image_dir + data[0])).unsqueeze(0).to(device)
            if avgpool is False:
                image_features = model.encode_image(images).squeeze().cpu().numpy()  # (196,512)
                total_feat.update({data[0]: image_features0})
            if avgpool is True:
                image_features = model.encode_image(images).squeeze().mean(dim=0, keepdim=True).cpu().numpy()  # (1,512)
                total_feat.update({data[0]: image_features})

np.savez_compressed(image_feat_path, feature=total_feat)
