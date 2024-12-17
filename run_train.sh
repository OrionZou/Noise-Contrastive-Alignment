export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tf/orion.zou/huggingface


export PYTHONPATH=.:$PYTHONPATH


# CUDA_VISIBLE_DEVICES=4,5  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port=7000 run_reward.py yamls/reward_qlora.yaml --gradient_accumulation_steps=4 --beta=0.01 --loss_type=NCA --output_dir=data/llama-3-8b-instruct-nca-r-2a800 > llama-3-8b-instruct-nca-r-2a800.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3,4,5  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port=7000 run_reward.py yamls/reward_qlora.yaml --gradient_accumulation_steps=16 --per_device_train_batch_size=2 --beta=0.01 --loss_type=InfoNCA --output_dir=data/llama-3-8b-instruct-infonca-r-2a800 > llama-3-8b-instruct-infonca-r-2a800.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3,4,5  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port=7000 run_reward.py yamls/reward_qlora.yaml --gradient_accumulation_steps=16 --per_device_train_batch_size=2 --beta=0.01 --loss_type=InfoNCA --output_dir=data/llama-3-8b-instruct-infonca-r-2a800 > llama-3-8b-instruct-infonca-r-2a800.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port=7000 run_preference.py yamls/preference_qlora.yaml --gradient_accumulation_steps=16 --beta=0.01 --loss_type=NCA --output_dir=data/llama-3-8b-instruct-nca-p-2a800 > llama-3-8b-instruct-nca-p-2a800.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port=7000 run_reward.py yamls/reward_qlora.yaml --learning_rate=0.000005 --gradient_accumulation_steps=32 --beta=0.01 --loss_type=NCA --output_dir=data/llama-3-8b-instruct-nca-r-2a800-2 > llama-3-8b-instruct-nca-r-2a800-2.log 2>&1 &
