export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=7

python tools/run_on_lasot_args.py --samsam2 --cfg configs/samsam2/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset test --out_dir results/test/samsam2_baseplus
# python tools/run_on_lasot_args.py --samsam2 --cfg configs/samsam2/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset extra --out_dir results/test/samsam2_baseplus