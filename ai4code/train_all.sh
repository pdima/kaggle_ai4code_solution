# edit config.py to specify the path to the train and optionally external dataset
# edit experiments/l2*.yaml to specify the path to the external dataset

python prepare_dataset.py
python detect_language.py

# train model, up to 770 epochs
python train.py train --fold 0 experiments/342_single_bert_l2_scaled_att.yaml
# predict the base transformers activations, to be used for L2 model training
python train.py predict_encoders_train --fold 0 experiments/342_single_bert_l2_scaled_att.yaml --epoch 770
# if external data is used, predict the base transformers activations, to be used for L2 model training
python train.py predict_extra --fold 0 experiments/342_single_bert_l2_scaled_att.yaml --epoch 770
# small extra step to save the empty code cell prediction
python train.py save_empty_code --fold 0 experiments/342_single_bert_l2_scaled_att.yaml --epoch 770
# export the trained model (remove optimiser parameters etc)
python train.py export --fold 0 experiments/342_single_bert_l2_scaled_att.yaml --epoch 770
# repeat the same steps for folds 1,2,3


python train.py train --fold 0 experiments/356_bert_mpnet_l2_madgrad.yaml
python train.py predict_encoders_train --fold 0 experiments/356_bert_mpnet_l2_madgrad.yaml --epoch 798
python train.py predict_extra --fold 0 experiments/356_bert_mpnet_l2_madgrad.yaml --epoch 798
python train.py save_empty_code --fold 0 experiments/356_bert_mpnet_l2_madgrad.yaml --epoch 798
python train.py export --fold 0 experiments/356_bert_mpnet_l2_madgrad.yaml --epoch 798
# repeat the same steps for folds 0,1,2


# train the separate l2 models
python train_l2.py train --fold 0 experiments/l2_500_l6_b64_w64.yaml
python train_l2.py train --fold 1 experiments/l2_501_l6_b64_w64.yaml
python train_l2.py train --fold 2 experiments/l2_502_l6_b64_w64.yaml
python train_l2.py train --fold 3 experiments/l2_503_l6_b64_w64.yaml

python train_l2.py train --fold 0 experiments/l2_510_l6_b64_w64.yaml
python train_l2.py train --fold 1 experiments/l2_511_l6_b64_w64.yaml
python train_l2.py train --fold 2 experiments/l2_512_l6_b64_w64.yaml

python train_l2.py export experiments/l2_700_l2_light.yaml --fold 0 --epoch 770

# export l2 models:
python train_l2.py export experiments/l2_500_l6_b64_w64.yaml --fold 0 --epoch 770
python train_l2.py export experiments/l2_501_l6_b64_w64.yaml --fold 1 --epoch 770
python train_l2.py export experiments/l2_502_l6_b64_w64.yaml --fold 2 --epoch 770
python train_l2.py export experiments/l2_503_l6_b64_w64.yaml --fold 3 --epoch 770
python train_l2.py export experiments/l2_510_l6_b64_w64.yaml --fold 0 --epoch 770
python train_l2.py export experiments/l2_511_l6_b64_w64.yaml --fold 1 --epoch 770
python train_l2.py export experiments/l2_512_l6_b64_w64.yaml --fold 2 --epoch 770

python train_l2.py export experiments/l2_700_l2_light.yaml --fold 0 --epoch 770



