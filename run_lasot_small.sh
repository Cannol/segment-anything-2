export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=7

python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_s.yaml --checkpoint checkpoints/sam2.1_hiera_small.pt --dataset test --out_dir results/test/baseline_small
python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_s.yaml --checkpoint checkpoints/sam2.1_hiera_small.pt --dataset extra --out_dir results/extra/baseline_small