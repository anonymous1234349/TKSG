# pip install transformers==4.35.2
# 运行：HF_ENDPOINT=https://hf-mirror.com python BiomedCLIP_extract_report.py
# 参考：https://github.com/mlfoundations/open_clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch
from urllib.request import urlopen
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune
# 读取json
json_file_path = f'/home2/hongfei/dataset/{dataset_name}/annotation.json'
report_feat_path = f'/home2/hongfei/dataset/{dataset_name}/BiomedCLIP_report.npz'
print('annotation path:', json_file_path)
print('image feats save path:', report_feat_path)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)
id_report = {}
for split in ['train','val','test']:
    data = json_data[split]
    for i in data:
        if dataset_name == 'iu-xray':
            id_report.update({i['id']:i['report']})
        if dataset_name == 'mimic-cxr':
            id_report.update({i['image_path'][0]:i['report']})

images = preprocess(Image.open('/home2/hongfei/dataset/iu_xray-finetune/images/CXR1000_IM-0003/0.png')).unsqueeze(0).to(device)
text_feats = {}
for report_id, report_text in tqdm(id_report.items()):
    with torch.no_grad():
        text_token = tokenizer(report_text).to(device)
        _, text_feature, _ = model(images, text_token)
        text_feature = text_feature.detach().cpu().numpy()
        text_feats.update({report_id : text_feature})

np.savez_compressed(report_feat_path, feature=text_feats)
