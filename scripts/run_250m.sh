# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=$2

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=29500 torchrun_main.py \
    --model_config configs/llama_250m.json \
    --seed 42 \
    --lr $learning_rates \
    --max_length 256 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 40000 \
    --warmup_steps 4000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --clip_gate_grad 0.0 \
    --run_name "250m_res_${norm_type}_lr${learning_rates}" \
    --save_dir "250m_res_${norm_type}_lr${learning_rates}"