import torch
import json
import numpy as np
from tqdm import tqdm
import clip
import re

def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
model, preprocess = clip.load("ViT-B/16", device=device)

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune
json_file_path = f'/dataset/{dataset_name}/annotation.json'
report_feat_path = f'/dataset/{dataset_name}/CLIP-vit-16_report.npz'
print('annotation path:', json_file_path)
print('image feats save path:', report_feat_path)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)
id_report = {}
for split in ['train', 'val', 'test']:
    data = json_data[split]
    for i in data:
        id_report.update({i['image_path'][0]: i['report']})

text_feats = {}
for report_id, report_text in tqdm(id_report.items()):
    tokens = clip.tokenize([report_text], context_length=77, truncate=True).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens).detach().cpu().numpy()
        text_feats.update({report_id: text_embeds})

np.savez_compressed(report_feat_path, feature=text_feats)