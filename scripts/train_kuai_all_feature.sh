GPU_DEVICE="0"
# data
TRAIN_DATA_NAME="kuai50k"
VAL_DATA_NAME="kuai50k"
STRIDE="2.5"
FILTER="10"
PADDING="0"


# Model
VIDEO_ENCODER="ViT"
AUDIO_ENCODER="AST"
############# Enhancement #############     AGG_MODULE=["None", "mlp", "transf"]
AGG_MODULE="transf"
WITH_CLS_TOKEN="0"
WITH_LAST_TOKEN="0"
TRM_AGG_SHARE="0"
ACT_AFTER_PROJ="0"
SA_Heads="8"
Align_dim="256"
TRM_DEPTH="1"

############# Matching #############     Fusion: VMR_FUSION=["XA-music", "NO", "XA-music-video"]
VMR_FUSION="XA-music"
LOSS="DS_loss"  # ["S", "D", "DS_sim", "DS_loss", "DS_feat", "DSboth"], default: "DS_loss"
F_MASK="1"
DS_loss_weight="1.0"

############# Detection #############     MML_FUS=["CA", "concat", "add"], default: "concat"
MML_FUS="concat"
############# DETR #############     MML_LOC=["detr", "regression"], default: "detr"
MML_LOC="detr"
FB_LABEL="01"
DETR_ENC_LAYERS="2"
DETR_DEC_LAYERS="6"
MOMENT_QUERY="video"  # ["video", "xpool", "music", "zero", "random"] "zero" == "random", default: "video"
NUM_MOMENT_QUERIES="1"
DEC_SA="0"
PRED_CENTER="0"


# Loss
L1_LOSS="1"
AUX_LOSS="1"
CONTRASTIVE_LOSS="1"
AUDIO_SHORT_CUT="0"
# Weight
RET_WEIGHT="1.0"
LOC_WEIGHT="1.0"

# Train
ret_LR="3e-4"
det_LR="3e-4"
EPOCHS="100"
TRAIN_BS="512"
VAL_BS="40"
TEMP="3e-2"


if [ "${LOSS}" = "S" ]; then
    vmr_LOSS="single"
elif [ "${LOSS}" = "D" ]; then
    vmr_LOSS="dual"
elif [ "${LOSS}" = "DS_sim" ]; then
    vmr_LOSS="dual_single_sim_fuse"
elif [ "${LOSS}" = "DS_loss" ]; then
    vmr_LOSS="dual_single_loss_fuse"
elif [ "${LOSS}" = "DS_feat" ]; then
    vmr_LOSS="dual_single_feature_fuse"
elif [ "${LOSS}" = "DSboth" ]; then
    vmr_LOSS="dual_single_oneloss"
else
    echo "ERROR: loss type not supported"
    exit
fi



NAME="train-UNI"
NAME="${NAME}_[${AGG_MODULE}-clsa${WITH_CLS_TOKEN}${WITH_LAST_TOKEN}${TRM_AGG_SHARE}${ACT_AFTER_PROJ}-dim${Align_dim}-dep${TRM_DEPTH}-head${SA_Heads}]"
NAME="${NAME}_[fus${VMR_FUSION}-mask${F_MASK}_loss${LOSS}-w${DS_loss_weight}]"
NAME="${NAME}_[${MML_FUS}-${MML_LOC}${DETR_ENC_LAYERS}${DETR_DEC_LAYERS}-decSA${DEC_SA}-LA${L1_LOSS}${AUX_LOSS}-contra${CONTRASTIVE_LOSS}${AUDIO_SHORT_CUT}-MQ${MOMENT_QUERY}-#Q${NUM_MOMENT_QUERIES}-center${PRED_CENTER}]"
NAME="${NAME}_ep${EPOCHS}-S${STRIDE}-lr${ret_LR}+${det_LR}_bs${TRAIN_BS}_temp${TEMP}_weight${RET_WEIGHT}-${LOC_WEIGHT}"
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python -m torch.distributed.launch --nproc_per_node=1 --master_port 1111${GPU_DEVICE} \
    train-MaDe.py --name ${NAME} \
    --do_train --do_eval \
    --train_data ${TRAIN_DATA_NAME} --val_data ${VAL_DATA_NAME} \
    --stride ${STRIDE} --filter ${FILTER} --padding ${PADDING} --max_m_duration 240 --max_v_frames 50 --num_moment_queries ${NUM_MOMENT_QUERIES} \
    --video_encoder_type ${VIDEO_ENCODER} --audio_encoder_type ${AUDIO_ENCODER} \
    --agg_module ${AGG_MODULE} \
    --video_transformer_depth ${TRM_DEPTH} --audio_transformer_depth ${TRM_DEPTH} --SA_temporal_heads ${SA_Heads} --transformer_is_share ${TRM_AGG_SHARE} --with_cls_token ${WITH_CLS_TOKEN} --with_act_after_proj ${ACT_AFTER_PROJ} --with_last_token ${WITH_LAST_TOKEN} \
    --dim_input ${Align_dim} \
    --mml_fusion ${MML_FUS} --mml_localization ${MML_LOC} \
    --vmr_fusion ${VMR_FUSION} --vmr_loss ${vmr_LOSS} --fusion_mask ${F_MASK} --dual_single_loss_weight ${DS_loss_weight} \
    --detr_enc_layers ${DETR_ENC_LAYERS} --detr_dec_layers ${DETR_DEC_LAYERS} --decoder_SA ${DEC_SA} --moment_query_type ${MOMENT_QUERY} --predict_center ${PRED_CENTER} \
    --temperature_init_value ${TEMP} \
    --l1_loss ${L1_LOSS} --aux_loss ${AUX_LOSS} --contrastive_align_loss ${CONTRASTIVE_LOSS} --audio_short_cut ${AUDIO_SHORT_CUT} \
    --ret_loss_weight ${RET_WEIGHT} --loc_loss_weight ${LOC_WEIGHT} \
    --batch_size_train ${TRAIN_BS} --batch_size_val ${VAL_BS} --num_workers 32 --epochs ${EPOCHS} \
    --matching_lr ${ret_LR} --detection_lr ${det_LR} --scheduler warmupcosine --warmup_rate 0.02 --decay_rate 0.9 --lr_update_rate 40 \
    --distance_type COS \
    --train_csv ./dataset/MGSV-EC/train_data.csv \
    --val_csv ./dataset/MGSV-EC/val_data.csv \
    --frozen_feature_path ./features/Kuai_feature \
    --output_dir ./logs \
    --num_display 10 --tb_writer 1 --save_model 1 --save_json 0