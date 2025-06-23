export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=7

# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset test --out_dir results/test/baseline_baseplus
# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset extra --out_dir results/extra/baseline_baseplus

# python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --out_dir results/test/thplus_baseplus --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset test
# python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --out_dir results/extra/thplus_baseplus --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset extra 



# --------------------
python tools/run_on_lasot_args_new.py --method thplus --th 2.0 --addon_ths 6.0,4.0,3.0 --out_dir results/mas4sam_test_lasot/bplus_th2_643 --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset test
python tools/run_on_lasot_args_new.py --method thplus --th 2.0 --addon_ths 6.0,4.0,3.0 --out_dir results/mas4sam_test_lasot/bplus_th2_643 --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --dataset extra 