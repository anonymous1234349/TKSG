import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(images_id, output, reports_ids, reports_masks, is_retrieval=False, concept_targets=None, predict_concept=None, is_topic=False, topic_targets=None, topic=None):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    if is_retrieval:
        criterion2 = nn.BCEWithLogitsLoss()
        concept_targets = concept_targets.float()  

        retrieval_loss = criterion2(input=predict_concept, target=concept_targets) 
        loss = loss +  retrieval_loss
    if is_topic:
        criterion3 = nn.BCEWithLogitsLoss()
        topic_targets = topic_targets.float()
        topic_loss = criterion3(input=topic, target=topic_targets)
        loss = loss + topic_loss
    return loss
