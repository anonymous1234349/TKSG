import numpy as np
import json
from tqdm import tqdm
import torch


dataset_name = 'mimic_cxr-512'  # iu_xray-finetune
model = 'CLIP-vit-b-16'  # Medclip/BiomedCLIP/CLIP-vit-b-16
topk = 100
batch_size = 4096  
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

json_file_path = f'/dataset/{dataset_name}/annotation.json'
images_feats_path = f"/dataset/{dataset_name}/{model}_image.npz"
reports_feats_path = f"/dataset/{dataset_name}/{model}_report.npz"
retrieval_base_path = f'/dataset/{dataset_name}/{model}_retrieval_batch_'  

images_data = np.load(images_feats_path, allow_pickle=True)
reports_data = np.load(reports_feats_path, allow_pickle=True)

images_feature = images_data['feature'].item()
reports_feature = reports_data['feature'].item()

with open(json_file_path, 'r') as f:
    json_data = json.load(f)

image_paths = []
for split in ['train', 'val', 'test']:
    data = json_data[split]
    image_paths.extend([i['image_path'] for i in data])

image_features_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in images_feature.items()}
reports_feature_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in reports_feature.items()}

all_report_features = torch.stack(list(reports_feature_tensor.values())).squeeze()  # (N_report, D)
print(f"{model} image and report feature loaded!")


def calculate_similarity(image_features, report_features):
    return torch.matmul(image_features, report_features.t())  


print("Start calculating similarity in batches...")
num_images = len(image_paths)
num_batches = (num_images + batch_size - 1) // batch_size  

for batch_idx in tqdm(range(num_batches)):
    batch_image_paths = image_paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_image_features = torch.stack([image_features_tensor[img_path[0]] for img_path in batch_image_paths]).squeeze()  # (batch_size, D)
    similarity_matrix = calculate_similarity(batch_image_features, all_report_features)  # (batch_size, N_report)

    reference = {}

    for idx, img_path_list in enumerate(batch_image_paths):
        img_path = img_path_list[0]  
        similarity_scores = similarity_matrix[idx].squeeze()
        sorted_indices = torch.argsort(similarity_scores, descending=True)[:topk]
        top_reports = [list(reports_feature_tensor.keys())[i] for i in sorted_indices]
        reference[img_path] = top_reports

    del batch_image_features, similarity_matrix
    torch.cuda.empty_cache()

    retrieval_path = f"{retrieval_base_path}{batch_idx}.npz"
    np.savez_compressed(retrieval_path, feature=reference)
    print(f"Batch {batch_idx} results saved to {retrieval_path}")