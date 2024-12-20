# 运行：HF_ENDPOINT=https://hf-mirror.com python MedCLIP_extract_report.py
# 参考：https://github.com/RyanWangZf/MedCLIP
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
import torch
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

# load MedCLIP-ResNet50
# model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
# model.from_pretrained()

# load MedCLIP-ViT
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.to(device)
model.from_pretrained()

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune
# 读取json
json_file_path = f'/home2/hongfei/dataset/{dataset_name}/annotation.json'
report_feat_path = f'/home2/hongfei/dataset/{dataset_name}/MedCLIP_report.npz'
print('annotation path:', json_file_path)
print('image feats save path:', report_feat_path)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)
id_report = {}
for split in ['train','val','test']:
    data = json_data[split]
    for i in data:
        id_report.update({i['image_path'][0]:i['report']})

text_feats = {}
for report_id, report_text in tqdm(id_report.items()):
    input = processor(images=Image.open("/home2/hongfei/dataset/iu_xray-finetune/images/CXR1000_IM-0003/0.png"), text=[report_text],return_tensors="pt").to(device)
    # images1 = processor(Image.open('/home/users/liuhf/Dataset/iu_xray/images/'+data[1]),return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**input)#.squeeze().cpu().numpy()
        text_embeds = output['text_embeds'].detach().cpu().numpy()  # (1, 512)
        text_feats.update({report_id:text_embeds})

np.savez_compressed(report_feat_path, feature=text_feats)