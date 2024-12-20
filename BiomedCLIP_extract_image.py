# pip install transformers==4.35.2 -i https://pypi.mirrors.ustc.edu.cn/simple
# 运行：HF_ENDPOINT=https://hf-mirror.com python BiomedCLIP_extract_image.py 
# 参考：https://github.com/mlfoundations/open_clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch
from urllib.request import urlopen
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune
json_file_path = f'/dataset/{dataset_name}/annotation.json'
image_dir = f'/dataset/{dataset_name}/images/'
image_feat_path = f'/dataset/{dataset_name}/BiomedCLIP_image.npz'

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

template = 'this is a photo of '
labels = [
    'bone X-ray',
    'chest X-ray',
]
context_length = 256
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
image_feat = {}
for data in tqdm(image_path):
    if dataset_name == 'mimic_cxr-512':
        images = preprocess(Image.open(image_dir + data[0])).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features, _, _ = model(images, texts)
            image_features = image_features.detach().cpu().numpy()
            image_feat.update({data[0]:image_features})

    if dataset_name == 'iu_xray-finetune':
        images0 = preprocess(Image.open(image_dir+data[0])).unsqueeze(0).to(device)
        images1 = preprocess(Image.open(image_dir+data[1])).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features0, _, _ = model(images0, texts)
            image_features0 = image_features0.detach().cpu().numpy()
            image_feat.update({data[0]:image_features0})
            image_features1, _, _ = model(images1, texts)
            image_features1 = image_features1.detach().cpu().numpy()
            image_feat.update({data[1]:image_features1})

np.savez_compressed(image_feat_path, feature=image_feat)