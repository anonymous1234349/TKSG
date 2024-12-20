import argparse
import numpy as np
import torch
from models.models import TKSGModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer

# swin transformer parameter:
# tiny:  47672170
# small: 68990074
# base: 107486120
# ViT parameter:  109247728
def parse_agrs():
    parser = argparse.ArgumentParser()
    # retrieval
    parser.add_argument('--topic', action='store_true', help='disease topic')
    parser.add_argument('--new_topic', action='store_true', help='disease topic')
    parser.add_argument('--topic_path', type=str,default='/data/iu_xray/topic_MIMIC_CXR.json', help='disease topic path')
    parser.add_argument('--report_topk', type=int, default=15, help='reference Numbers')  
    parser.add_argument('--concept_num', type=int, default=100, help='Concept Numbers')
    parser.add_argument('--concept_topk', type=int, default=20, help='keyword Topk')
    parser.add_argument('--concept_dim', type=int, default=2048, help='Concept dimension')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')  

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.') 
    parser.add_argument('--resblocks_num', type=float, default=4,
                        help='finetne clip resblocks num.')
    parser.add_argument('--lr_ve', type=float, default=2e-4, 
                        help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=5e-4, 
                        help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-05, 
                        help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-09, 
                        help='the weight decay.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    parser.add_argument('--device', type=str, default='cuda:0', help='clip')
    parser.add_argument('--topk', type=int, default=24, help='the number of k.') 
    parser.add_argument('--clip', type=str, default='ViT-B/16', help='clip frame') 
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')  
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')

    # model frame
    parser.add_argument('--model_frame', type=str, default='swin_transformer_tiny',
                        choices=['ResNet','clip_feat', 'finetune_clip', 'swin_transformer_base', 'swin_transformer_tiny', 'swin_transformer_small', 'ViT']) 
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--global_guide', action='store_true')
    parser.add_argument('--local_guide', action='store_true')
    parser.add_argument('--segment', action='store_true')

    # feat
    parser.add_argument('--images_feat_path', type=str,
                        default='/dataset/iu_xray-finetune/BiomedCLIP_image.npz',
                        help='reference_images_feat_path')  # image feat
    parser.add_argument('--reports_feat_path', type=str,
                        default='/dataset/iu_xray-finetune/BiomedCLIP_report.npz',
                        help='reference_report_feat_path')  # report feat
    parser.add_argument('--reference_report_path', type=str,
                        default='/dataset/iu_xray-finetune/BiomedCLIP_retrieval_100.npz',
                        help='reference_report_path')  # reference report

    parser.add_argument('--clip_feat_path', type=str, 
                        default='/dataset/iu_xray-finetune/CLIP_image.npz',
                        help='clip feat')  # clip extract feat
    parser.add_argument('--vocab_path', type=str, default='/data/', help='vocab path.') 
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/dataset/iu_xray-finetune/images',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/dataset/iu_xray-finetune/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')

    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='/results', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=50, help='the logging interval (in batches).')  #1000
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=10, help='the step size of the learning rate scheduler.') # 50

    # Others
    parser.add_argument('--seed', type=int, default=7580, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

def print_args(args):
    if args.dataset_name == 'mimic_cxr':
        args.images_feat_path = '/dataset/mimic_cxr-512/BiomedCLIP_image.npz'
        args.reports_feat_path = '/dataset/mimic_cxr-512/BiomedCLIP_report.npz'
        args.reference_report_path = '/dataset/mimic_cxr-512/BiomedCLIP_retrieval.npz'
        args.topic_path = '/data/mimic_cxr/topic_MIMIC_CXR.json'
    else: # iu_xray
        args.images_feat_path = '/dataset/iu_xray-finetune/CLIP_image(1×512)-vit-b-16.npz'
        args.reports_feat_path = '/dataset/iu_xray-finetune/CLIP_report-vit-b-16.npz'
        args.reference_report_path = '/dataset/iu_xray-finetune/CLIP-vit-b-16_retrieval_100.npz'
        args.topic_path = '/data/iu_xray/topic_IU_Xray.json'

    print("############################# basic config #############################")
    print('dataset_name:', args.dataset_name)
    print('device:', args.device)
    print('model frame:', args.model_frame)
    print('clip:', args.clip)
    print('seed:', args.seed)
    print('batch_size:', args.batch_size)
    print('topk:', args.topk)
    print('optim: ', args.optim)
    print('lr_ve: ', args.lr_ve)
    print('lr_ed:', args.lr_ed)
    print('weight_decay:', args.weight_decay)
    print('adam_eps:', args.adam_eps)
    print('resblocks_num:',args.resblocks_num)
    print('save_dir:', args.save_dir)
    print("############################# retrieval #############################")
    if args.retrieval:
        print('retrieval:', args.retrieval)
        print('images_feat_path:', args.images_feat_path)
        print('reports_feat_path:', args.reports_feat_path)
        print('reference_report_path:',args.reference_report_path)
        print('local_guide:', args.local_guide)
        print('global_guide:', args.global_guide)
        print('report_topk:', args.report_topk * 2)
        print('concept_num:', args.concept_num)
        print('concept_topk:', args.concept_topk)
    print("############################# topic #############################")
    if args.topic:
        print('topic:', args.topic)
        print('topic path:', args.topic_path)
    if args.new_topic:
        print('new_topic:', args.new_topic)
        print('topic path:', args.topic_path)

def main():
    # parse arguments
    args = parse_agrs()
    print_args(args)
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    concept_targets = None
    if hasattr(tokenizer,'concept_targets'):
        concept_targets = tokenizer.concept_targets
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, concept_targets=concept_targets)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = TKSGModel(args, tokenizer)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()