python main_train.py \
  --image_dir /dataset/mimic_cxr-512/images/ \
  --ann_path /dataset/mimic_cxr-512/annotation.json \
  --dataset_name mimic_cxr \
  --max_seq_length 100 \
  --threshold 10 \
  --epochs 30 \
  --step_size 3 \
  --seed 9153 \
  --save_dir /results/best_MIMIC/ \
  --log_period 1000 \
  --device cuda:0 \
  --lr_ve 2e-4 \
  --lr_ed 5e-4 \
  --model_frame swin_transformer_tiny \
  --topk 24 \
  --new_topic \
  --retrieval \
  --local_guide \
  --report_topk 30 \
  --concept_num 100 \
  --concept_topk 20