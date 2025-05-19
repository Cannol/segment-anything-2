"""
run experiments with LaSOT and LaSOText dataset

"""

import os
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

LaSOTDataset_Conf = {
    "test":{
        "home": "/data/LaSOT/LaSOTTest/LaSOTTest",
        "list_file": "list.txt"
    },
    "extra":{
        "home":"/data/LaSOT/LaSOText",
        "list_file": "list.txt"
    }
}

def get_txt_list(list_file, parse_int=False, index=-1):
    with open(list_file) as f:
        lines = f.readlines()
    if index >= 0:
        lines = [lines[index]]
    if not parse_int:
        return [v.strip() for v in lines]
    else:
        return [list(map(int, v.strip().split(','))) for v in lines]

class LaSOTDataset(object):
        
    def __init__(self, dataset_name, subset_indexes=[], subset_rate_range=None, seqname_list=None, class_filter=None):
        """
        subset_indexes: list[int]
        subset_rate_range: [start_index_rate, end_index_rate)
        seqname_list: [str]
        class_filter: str / list[str]
        """
        self.conf = LaSOTDataset_Conf.get(dataset_name, None)
        self.conf['seq_all_names'] = get_txt_list(os.path.join(self.conf['home'], self.conf['list_file']))
        self._length_all = len(self.conf['seq_all_names'])
        # self.seqname_list = []

        if self.conf is None:
            raise ValueError(f'Unrecognized subset name of LaSOT: {dataset_name}')
        
        if len(subset_indexes) > 0:
            self.seqname_list = [self.conf['seq_all_names'][i] for i in subset_indexes]
        elif subset_rate_range is not None:
            start_index = int(self._length_all*subset_indexes[0])
            end_index = int(self._length_all*subset_indexes[1])
            self.seqname_list = [self.conf['seq_all_names'][i] for i in range(start_index, end_index)]
        elif seqname_list is not None:
            self.seqname_list = [i for i in seqname_list]
        else:
            # use all data
            self.seqname_list = self.conf['seq_all_names']
        
        # filtering
        if isinstance(class_filter, str):
            class_filter = [class_filter]
            self.seqname_list = [i for i in self.seqname_list if i.split['-'][0] in class_filter]
    
    def __len__(self): return len(self.seqname_list)
        
    def get_seq_reader(self, seq_index, output_format="path"):
        """
        output_format: "path", "numpy_bgr", "image"
        """
        seqname = self.seqname_list[seq_index]
        
        seq_root = os.path.join(self.conf['home'], seqname)
        imgs_root = os.path.join(seq_root, 'img')
        frame_jpgs = os.listdir(imgs_root)
        frame_jpgs.sort()

        if output_format == 'path':
            for i, frame_file in enumerate(frame_jpgs):
                yield i, os.path.join(imgs_root, frame_file)
    
    def get_seq_img_root(self, seq_index):
        return os.path.join(self.conf['home'], self.seqname_list[seq_index], 'img')
    
    
    def get_initial_rect(self, seq_index):
        gt_file = os.path.join(self.conf['home'], self.seqname_list[seq_index], 'groundtruth.txt')
        return get_txt_list(gt_file,parse_int=True,index=0)
        
def mask2bbox(mask):
    nonzero_indices = np.nonzero(mask)  # 获取非零值的索引
    if len(nonzero_indices[0]) > 0:
        # print(nonzero_indices)
        min_y, min_x = np.min(nonzero_indices, axis=1)  # 计算最小的 y 和 x 坐标
        max_y, max_x = np.max(nonzero_indices, axis=1)  # 计算最大的 y 和 x 坐标
        bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
        return bbox
    return [-10, -10, -1, -1]

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='results/test/baseline_base_plus',
        help="directory containing input masks (as PNG files) of each video",
    )

    return parser.parse_args()

def cal_near(df_new):
    neighbours = NearestNeighbors(n_neighbors=4)
    nbrs = neighbours.fit(df_new)
    dis, ind = nbrs.kneighbors(df_new)
    dis = np.sort(dis, axis=0)
    dis = dis[:, -1]
    plt.figure(figsize=(8, 8))
    plt.plot(dis)
    plt.show()

def get_clusters(data):
    data = np.stack(data, axis=1).squeeze()
    # scale = StandardScaler()
    # normal = MinMaxScaler()
    pca = PCA(n_components=2)
    data_pca = data.transpose()
    # features = scale.fit_transform(data)
    # features = normal.fit_transform(features)
    data = pca.fit_transform(data_pca)
    data = pd.DataFrame(data, columns=['X', 'Y'])
    data.to_csv("data1.csv", index=False)
    cluster = DBSCAN(eps=0.1, min_samples=50)
    data['Label'] = cluster.fit_predict(data)
    plt.figure(figsize=(15, 12))
    sns.scatterplot(x='X', y='Y', data=data, hue='Label', palette='tab10', s=200)
    sns.lineplot(x='X',
                 y='Y',
                 data=data
                 )
    # plt.plot(data_pca[:,0], data_pca[:,1])
    plt.show()

@torch.inference_mode()
def main():
    from sam2.build_sam import build_sam2_video_predictor

    configs = read_args()
    sam2_checkpoint = configs.checkpoint
    model_cfg = configs.cfg
    dataset_name = configs.dataset
    result_dir = configs.out_dir
    os.makedirs(result_dir, exist_ok=True)


    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    dataset = LaSOTDataset(dataset_name,seqname_list=['airplane-13'])
    for i, seq in enumerate(dataset.seqname_list):
        
        # results_save_file = os.path.join(result_dir, f"{seq}.txt")
        print(f"[{i+1}/{len(dataset)}] Processing...{seq}")

        inference_state = predictor.init_state(video_path=dataset.get_seq_img_root(i),
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=True,
                                               async_loading_frames=True)
        device_inside = inference_state["device"]

        nn = torch.nn.AvgPool1d(256)
        vis_data = []
        for frame_index in range(inference_state["num_frames"]):
            image = inference_state["images"][frame_index].to(device_inside).float().unsqueeze(0)
            backbone_out = predictor.forward_image(image)
            # print(backbone_out.shape)
            visf = backbone_out["vision_features"]
            # b, c, h, w -> b, h, w, c
            visf = visf.permute(0, 2, 3, 1)
            visf = visf.reshape(1, -1, 256)
            visf = nn(visf)
            visf = visf.detach().cpu().numpy().squeeze().reshape(visf.shape[1], -1)
            vis_data.append(visf)
            if frame_index > 0 and frame_index % 100 == 0:
                print(f"[progress: {frame_index+1}/{inference_state['num_frames']}]")

        get_clusters(vis_data)
        exit()



        # for features in predictor.propagate_in_video(inference_state):
        #     out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        #     # print(out_mask.shape)
        #     bbox = mask2bbox(out_mask[0])
        #     x, y, w, h = bbox
        #     results.append(f"{x},{y},{w},{h}")
        #
        # results = '\n'.join(results)
        #
        # with open(results_save_file, 'w') as f: f.write(results)
        predictor.reset_state(inference_state)

if __name__ == "__main__":
    main()