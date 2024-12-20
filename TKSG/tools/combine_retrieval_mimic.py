import numpy as np
from tqdm import tqdm

model = "BiomedCLIP" # MedCLIP, CLIP
file_dir = f"/dataset/mimic_cxr-512/{model}_retrieval_batch_"
save_path = f"/dataset/mimic_cxr-512/{model}_retrieval.npz"

# data = np.load(save_path,allow_pickle=True)
# data = data['feature'].item()


merged_data = {}
for i in tqdm(range(68)):
    file_name = file_dir + f"{i}.npz"
    data = np.load(file_name,allow_pickle=True)
    data = data['feature'].item()
    for key, value in data.items():
        if key not in merged_data:
            merged_data[key] = value

print(len(merged_data)) # 276778


np.savez_compressed(save_path, feature=merged_data)
print(f"data has save in: {save_path}")