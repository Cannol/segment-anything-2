export export PYTHONPATH=/home/lyx/codes/SAMSAM2
export CUDA_VISIBLE_DEVICES=6

# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/test/baseline_large
# python tools/run_on_lasot_args.py --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/extra/baseline_large

python tools/run_on_lasot_args.py --threshold 0 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th0
python tools/run_on_lasot_args.py --threshold 0 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th0

python tools/run_on_lasot_args.py --threshold 2 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th2
python tools/run_on_lasot_args.py --threshold 2 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th2

python tools/run_on_lasot_args.py --threshold 3 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th3
python tools/run_on_lasot_args.py --threshold 3 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th3

python tools/run_on_lasot_args.py --threshold 4 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th4
python tools/run_on_lasot_args.py --threshold 4 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th4

python tools/run_on_lasot_args.py --threshold 6 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th6
python tools/run_on_lasot_args.py --threshold 6 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th6

python tools/run_on_lasot_args.py --threshold 8 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th8
python tools/run_on_lasot_args.py --threshold 8 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th8

python tools/run_on_lasot_args.py --threshold 10 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th10
python tools/run_on_lasot_args.py --threshold 10 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th10

python tools/run_on_lasot_args.py --threshold 5 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th5
python tools/run_on_lasot_args.py --threshold 5 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th5

python tools/run_on_lasot_args.py --threshold 7 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th7
python tools/run_on_lasot_args.py --threshold 7 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th7

python tools/run_on_lasot_args.py --threshold 9 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset test --out_dir results/lasot/baseline_large_th9
python tools/run_on_lasot_args.py --threshold 9 --cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint checkpoints/sam2.1_hiera_large.pt --dataset extra --out_dir results/lasot/baseline_large_th9