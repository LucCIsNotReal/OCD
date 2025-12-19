PYTHON='python3'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
# Save outputs under project-local outputs folder
SAVE_DIR=/home/bao/bao/OCD/On-the-fly-Category-Discovery/outputs/cub/

# Ensure save dir exists
mkdir -p "${SAVE_DIR}"

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m train_smile \
            --dataset_name 'cub' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 100 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.5 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.01 \
            --eval_funcs 'v1' 'v2' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out
