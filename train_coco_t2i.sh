MODEL_NAME='coco_t2i_base'
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 5 \
 --attn_type t2i --t2i_smooth 10.0 --i2t_smooth 3.0 \
 --self_regulator only_rcr --rcar_step 0 --rcr_step 0 --rar_step 0


MODEL_NAME='coco_t2i_rar3'
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 5 \
 --attn_type t2i --t2i_smooth 10.0 --i2t_smooth 3.0 \
 --self_regulator only_rar --rcar_step 0 --rcr_step 0 --rar_step 3


MODEL_NAME='coco_t2i_rcr2'
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 5 \
 --attn_type t2i --t2i_smooth 10.0 --i2t_smooth 3.0 \
 --self_regulator only_rcr --rcar_step 0 --rcr_step 2 --rar_step 0


MODEL_NAME='coco_t2i_rcar2'
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
 --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
 --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES}\
 --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 10 \
 --log_step 100 --embed_size 1024 --vse_mean_warmup_epochs 5 \
 --attn_type t2i --t2i_smooth 10.0 --i2t_smooth 3.0 \
 --self_regulator coop_rcar --rcar_step 2 --rcr_step 0 --rar_step 0