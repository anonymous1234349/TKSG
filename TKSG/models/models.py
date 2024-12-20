import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.retrieval import Retrieval
from modules.mask_encoder import MaskViTFeatureExtractor

class TKSGModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(TKSGModel, self).__init__()
        self.args = args
        self.dataset_name = args.dataset_name
        self.retrieval = args.retrieval  # use retrieval?
        self.new_topic = args.new_topic
        self.topic = args.topic
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)

        if self.retrieval:
            self.retrieval_model = Retrieval(args, 512,512, args.d_model, tokenizer.concepts_dict)
            for module in self.retrieval_model.modules(): 
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.topic:
            if self.dataset_name == 'mimic_cxr':
                self.topic_predict_0 = nn.Linear(2048, 14)
                self.topic_linear = nn.Linear(14, 512)
            if self.dataset_name == 'iu_xray':
                self.topic_predict_0 = nn.Linear(4096, 2048)
                self.topic_predict_1 = nn.Linear(2048, 1024)
                self.topic_predict_2 = nn.Linear(1024, 512)
                self.topic_predict_3 = nn.Linear(512, 14)
                self.topic_linear = nn.Linear(14, 512)
                nn.init.xavier_uniform_(self.topic_predict_0.weight)
                nn.init.xavier_uniform_(self.topic_predict_1.weight)
                nn.init.xavier_uniform_(self.topic_predict_2.weight)
                nn.init.xavier_uniform_(self.topic_predict_3.weight)
                nn.init.xavier_uniform_(self.topic_linear.weight)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, retrieval_image_feat, retrieval_report_feat, mask, targets=None, mode='train', update_opts={}):
        global_guide, local_guide, probabilities, topic_probabilities, topic = None, None, None, None, None

        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if self.topic: 
            topic_output = self.topic_predict_0(fc_feats)
            topic_output = self.topic_predict_1(topic_output) 
            topic_output = self.topic_predict_2(topic_output)
            topic_output = self.topic_predict_3(topic_output)
            topic_probabilities = torch.sigmoid(topic_output)

            topic = self.topic_linear(topic_probabilities)

        if self.retrieval:  # use retrieval ?
            retrieval_image_feat = att_feats.mean(dim=1)
            global_guide, local_guide, probabilities = self.retrieval_model(retrieval_image_feat, retrieval_report_feat)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', global_guide=global_guide, local_guide=local_guide, topic=topic)
            return output, probabilities, topic_probabilities
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, global_guide=global_guide, local_guide=local_guide, topic=topic)
            return output, output_probs, probabilities  
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, retrieval_image_feat, retrieval_report_feat, mask, targets = None, mode = 'train', update_opts = {}):
        global_guide, local_guide, probabilities, topic_probabilities, topic = None, None, None, None, None


        att_feats, fc_feats = self.visual_extractor(images)

        if self.retrieval:  # use retrieval ?
            retrieval_image_feat = att_feats.mean(dim=1)
            global_guide, local_guide, probabilities = self.retrieval_model(retrieval_image_feat, retrieval_report_feat)

        if self.topic: 
            topic_output = self.topic_predict_0(fc_feats) 
            topic_probabilities = torch.sigmoid(topic_output)
            topic = self.topic_linear(topic_probabilities)

        if mode == 'train':
            output, topic_probabilities_new = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward', global_guide=global_guide, local_guide=local_guide, topic=topic)
            if self.new_topic:
                topic_probabilities = topic_probabilities_new
            return output, probabilities, topic_probabilities
        elif mode == 'sample':
            output, output_probs, topic_probabilities_new = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, global_guide=global_guide, local_guide=local_guide, topic=topic)
            if self.new_topic:
                topic_probabilities = topic_probabilities_new
            return output, output_probs, probabilities, topic_probabilities
        else:
            raise ValueError
