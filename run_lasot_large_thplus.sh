export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=7

# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/test/baseline_large
# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/extra/baseline_large

python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --out_dir results/test/thplus_large --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test
# python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --out_dir results/extra/thplus_large --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra