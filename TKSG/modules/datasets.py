import json
import os
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle
from torch.utils.data import Dataset
import numpy as np
from CLIP import clip
# transform
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, concept_targets=None):
        self.model_frame = args.model_frame  #
        self.retrieval = args.retrieval  # use retrieval?
        self.image_dir = args.image_dir
        self.report_topk = args.report_topk  # reference report numbers
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length  # 60
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.concept_targets = concept_targets
        # /data/mimic_cxr_report_target.json
        self.target_path = args.vocab_path + args.dataset_name + '_report_target.json' 
        if not os.path.isfile(self.target_path):  
            ids = {}
            for s in ['train', 'val', 'test']:
                tmp = self.ann[s]
                for i in tqdm(range(len(tmp))):
                    ids[tmp[i]['id']] = tokenizer(tmp[i]['report'])[:self.max_seq_length]

            with open(self.target_path, 'w') as f: 
                json.dump({'ids':ids}, f)
                print(f'ids save in {self.target_path}.')

        with open(self.target_path, 'r') as f: 
            self.report_target = json.load(f)

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = self.report_target['ids'][self.examples[i]['id']]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

        ####################################### read raw CLIP extract image feat #########################################
        if self.model_frame == 'clip_feat':
            if args.dataset_name == "iu_xray":
                data = np.load(args.clip_feat_path, allow_pickle=True)
                self.features = data['feature'].item()[self.split]
        ####################################### finetune CLIP ##################################################
        if self.model_frame in ['finetune_clip', 'swin_transformer_tiny']:
            _, self.preprocess = clip.load(args.clip, device=args.device)
        ####################################### swin_transformer / ViT ##################################################
        if self.model_frame in [ 'swin_transformer_base', 'swin_transformer_small', 'ViT']:
            self.preprocess = transforms.Compose([
                                transforms.Resize(224, interpolation=Image.BICUBIC, antialias=True),
                                transforms.CenterCrop(224),
                                transforms.Lambda(lambda image: image.convert('RGB')),  # 等效于 _convert_image_to_rgb
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                            ])
        ###################################### load retrieval data #############################################
        if self.retrieval:
            if args.dataset_name == "mimic_cxr":
                data = np.load(args.images_feat_path, allow_pickle=True)  # image feat
                self.images_feat = data['feature'].item()
                data = np.load(args.reports_feat_path, allow_pickle=True)  # report feat
                self.report_feat = data['feature'].item()
                data = np.load(args.reference_report_path, allow_pickle=True)  # reference report
                self.reference_report = data['feature'].item()
            else:  # iu_xray
                data = np.load(args.images_feat_path, allow_pickle=True)  # image feat
                self.images_feat = data['feature'].item()
                data = np.load(args.reports_feat_path, allow_pickle=True)  # report feat
                self.report_feat = data['feature'].item()
                data = np.load(args.reference_report_path, allow_pickle=True)  # reference report
                self.reference_report = data['feature'].item()
        ###################################### topic  #############################################
        self.topic = args.topic
        if self.topic:
            self.topic_path = args.topic_path
            if args.dataset_name == "mimic_cxr":
                with open(self.topic_path, 'r') as json_file:
                    self.topic_data = json.load(json_file)
            else:  # iu_xray
                with open(self.topic_path, 'r') as json_file:
                    self.topic_data = json.load(json_file)
        ###################################### segment mask #############################################
        self.segment = args.segment
        if self.segment:
            self.mask_transforms = Compose([
                Resize(size=224, interpolation=InterpolationMode.NEAREST),  
                CenterCrop(size=(224, 224)),
                ToTensor() 
            ])
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        if self.model_frame == 'ResNet':
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)

        ##################################### read raw CLIP extract image feat ######################################
        if self.model_frame == 'clip_feat':
            image_1 = torch.tensor(self.features[image_path[0]], dtype=torch.float32)
            image_2 = torch.tensor(self.features[image_path[1]], dtype=torch.float32)
        ##################################### finetune CLIP ########################################
        if self.model_frame in ['finetune_clip', 'swin_transformer_base', 'swin_transformer_tiny', 'swin_transformer_small', 'ViT']:
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            image_1 = self.preprocess(image_1)  
            image_2 = self.preprocess(image_2)  
        ###################################### segment mask #############################################
        mask = None
        if self.segment:
            mask_0 = Image.open(os.path.join(self.image_dir, image_id, '0_s.png')).convert("L")
            mask_1 = Image.open(os.path.join(self.image_dir, image_id, '1_s.png')).convert("L")

            mask_0 = self.mask_transforms(mask_0)  
            mask_1 = self.mask_transforms(mask_1)  
            mask = torch.stack((mask_0, mask_1), 0)
        ###################################### load retrieval data #############################################
        retrieval_image_feat, retrieval_report_feat, concept_targets, topic_targets = None, None, None, None
        if self.retrieval:
            image_feat0 = torch.tensor(self.images_feat[image_path[0]], dtype=torch.float32)  # image0
            image_feat1 = torch.tensor(self.images_feat[image_path[1]], dtype=torch.float32)  # image1
            reference_report0 = self.reference_report[image_path[0]][: self.report_topk]  # reference0
            reference_report1 = self.reference_report[image_path[1]][:self.report_topk]  # reference1


            report_feat0_list, report_feat1_list, image_feat_list, report_feat_list  = [], [], [], []
            for i in reference_report0:
                report_feat_list.append(torch.tensor(self.report_feat[i], dtype=torch.float32))
                image_feat_list.append(torch.tensor(self.images_feat[i + '/0.png'], dtype=torch.float32))

            for j in reference_report1:
                report_feat_list.append(torch.tensor(self.report_feat[j], dtype=torch.float32))
                image_feat_list.append(torch.tensor(self.images_feat[j + '/0.png'], dtype=torch.float32))

            retrieval_report_feat = torch.cat(report_feat_list, dim=0)
            retrieval_image_feat = torch.cat([image_feat0, image_feat1], dim=0)

            if self.concept_targets is not None:
                concept_targets = torch.tensor(self.concept_targets[image_id])

        image = torch.stack((image_1, image_2), 0)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        if self.topic:
            topic_targets = torch.tensor(self.topic_data[example['id']])
        sample = (image_id, image, report_ids, report_masks, seq_length, retrieval_image_feat, retrieval_report_feat, mask, concept_targets, topic_targets)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):

        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_id = os.path.join(self.image_dir, image_path[0])
        image = Image.open(image_id).convert('RGB')

        if self.model_frame == 'ResNet':
            image = self.transform(image)
        if self.model_frame in ['finetune_clip', 'swin_transformer_base', 'swin_transformer_tiny', 'swin_transformer_small', 'ViT']:
            image = self.preprocess(image) # transform

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        retrieval_image_feat, retrieval_report_feat, mask, concept_targets, topic_targets = None, None, None, None, None
        if self.retrieval:
            # image_feat = torch.tensor(self.images_feat[image_path[0]], dtype=torch.float32)  # image0
            reference_report = self.reference_report[image_path[0]][:self.report_topk * 2]  # reference0

            report_feat_list, image_feat_list = [], []
            for i in reference_report:
                report_feat_list.append(torch.tensor(self.report_feat[i], dtype=torch.float32))
                image_feat_list.append(torch.tensor(self.images_feat[i], dtype=torch.float32))

            report_feat = torch.cat(report_feat_list, dim=0)
            image_feat = torch.cat(image_feat_list, dim=0)

            retrieval_image_feat = image_feat # torch.cat((image_feat0, image_feat1), dim=0)
            retrieval_report_feat = report_feat # torch.cat((report_feat0, report_feat1), dim=0)

            if self.concept_targets is not None:
                concept_targets = torch.tensor(self.concept_targets[image_path[0]])
        if self.topic:
            topic_targets = torch.tensor(self.topic_data[example['id']])
        sample = (image_id, image, report_ids, report_masks, seq_length, retrieval_image_feat, retrieval_report_feat, mask, concept_targets, topic_targets)
        return sample