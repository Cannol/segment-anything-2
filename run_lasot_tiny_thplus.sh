export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=6

# python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --cfg configs/sam2.1/sam2.1_hiera_t.yaml --checkpoint checkpoints/sam2.1_hiera_tiny.pt --dataset test --out_dir results/test/thplus_tiny
# python tools/run_on_lasot_args_new.py --method thplus --th 1.0 --cfg configs/sam2.1/sam2.1_hiera_t.yaml --checkpoint checkpoints/sam2.1_hiera_tiny.pt --dataset extra --out_dir results/extra/thplus_tiny

python tools/run_on_lasot_args_new.py --method thplus --th 2.0 --addon_ths 6.0,4.0,3.0 --out_dir results/mas4sam_test_lasot/tiny_th2_643 --cfg configs/sam2.1/sam2.1_hiera_t.yaml --checkpoint checkpoints/sam2.1_hiera_tiny.pt --dataset test 
python tools/run_on_lasot_args_new.py --method thplus --th 2.0 --addon_ths 6.0,4.0,3.0 --out_dir results/mas4sam_test_lasot/tiny_th2_643 --cfg configs/sam2.1/sam2.1_hiera_t.yaml --checkpoint checkpoints/sam2.1_hiera_tiny.pt --dataset extra