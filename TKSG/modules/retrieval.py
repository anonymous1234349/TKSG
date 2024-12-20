import torch
import torch.nn as nn
from modules.base_cmn import Embeddings, PositionalEncoding
import torch.nn.functional as F

class Retrieval(nn.Module):
    def __init__(self, args, input_dim, output_dim, d_model, concept_dict):
        super(Retrieval, self).__init__()
        self.args = args
        self.global_guide = args.global_guide
        self.concept_topk = args.concept_topk
        self.concept_num = args.concept_num
        self.image_encoder = nn.Linear(2048, output_dim)
        self.report_encoder = nn.Linear(input_dim, output_dim)

        self.concept_dector = nn.Linear(2*output_dim,  self.concept_num) # 新加
        # self.linear = nn.Linear(512, self.concept_num)

        if self.global_guide:
            self.global_linear = nn.Linear(self.concept_num, d_model)
        #
        # self.local_embedding = nn.Sequential(Embeddings(d_model=512, vocab=self.concept_num), PositionalEncoding(d_model=512, dropout=0.1))
        self.concept_dict = concept_dict

    def forward(self, retrieval_image_feat, retrieval_report_feat):

        image_feat = self.image_encoder(retrieval_image_feat)  # encoder
        report_feat = self.report_encoder(retrieval_report_feat)  # encoder
        if len(image_feat.shape) == 3:
            image_feat = image_feat.mean(dim=1, keepdim=True).squeeze()
        report_feat = report_feat.mean(dim=1, keepdim=True).squeeze()
        retrieval_feat = torch.cat((image_feat, report_feat), dim=-1)

        retrieval_feat = F.dropout(retrieval_feat, p=0.5, training=self.training)
        # retrieval_feat = F.layer_norm(retrieval_feat, retrieval_feat.size())

        output = self.concept_dector(retrieval_feat)
        # output = self.linear(tmp) 

        probabilities = torch.sigmoid(output)

        # raw = torch.log(torch.clamp(1.0 - probabilities, 1e-12, 1))  
        # outputs = 1.0 - torch.exp(raw)  

        global_guide = None
        if self.global_guide:
            global_guide = self.global_linear(probabilities)  # retrieval_feat
        value, local_index = torch.topk(probabilities, k=self.concept_topk, dim=1, largest=True, sorted=True)  # 选出前topk个单词

        # local_guide = self.local_embedding(local_index)

        local_index = local_index.to('cpu').tolist()
        local_word = []
        for b in local_index:
            tmp = []
            for i in b:
                tmp.append(self.concept_dict[i])
            local_word.append(tmp)
        local_word = torch.tensor(local_word).to(retrieval_image_feat.device)
        # local_word = local_guide
        return global_guide, local_word, probabilities