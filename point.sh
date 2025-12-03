   CUDA_VISIBLE_DEVICES=5 python visualize_gr.py \
     --model_path /mnt/netdisk2/zhangyf/model/FLUX.1-dev \
     --height 1024 --width 1024 --guidance_scale 3.5 --steps 50 \
     --init_mode 3 --gpu_ids 5 --disable_generation 