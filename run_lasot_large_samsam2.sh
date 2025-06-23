export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=6

python tools/run_on_lasot_args.py --samsam2 --cfg configs/samsam2/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/test/samsam2_large
python tools/run_on_lasot_args.py --samsam2 --cfg configs/samsam2/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/extra/samsam2_large