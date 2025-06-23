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
import cv2
from PIL import Image

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
        "home": "/data2/lyx/LaSOT/LaSOTTest",
        "list_file": "list.txt"
    },
    "extra":{
        "home":"/data2/lyx/LaSOT/LaSOText",
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

    def get_seq_frames(self, seq_index, output_format="path"):
        """
        output_format: "path", "numpy_bgr", "image"
        """
        seqname = self.seqname_list[seq_index]

        seq_root = os.path.join(self.conf['home'], seqname)
        imgs_root = os.path.join(seq_root, 'img')
        frame_jpgs = [i for i in os.listdir(imgs_root) if i.endswith('.jpg')]
        frame_jpgs.sort()

        if output_format == 'path':
            frames = []
            for i, frame_file in enumerate(frame_jpgs):
                frames.append(os.path.join(imgs_root, frame_file))
            return frames

        return None
    
    def get_seq_img_root(self, seq_index):
        return os.path.join(self.conf['home'], self.seqname_list[seq_index], 'img')
    
    
    def get_initial_rect(self, seq_index):
        gt_file = os.path.join(self.conf['home'], self.seqname_list[seq_index], 'groundtruth.txt')
        return get_txt_list(gt_file,parse_int=True,index=0)
        
def mask2bbox(mask, xywh=False):
    nonzero_indices = np.nonzero(mask)  # 获取非零值的索引
    if len(nonzero_indices[0]) > 0:
        # print(nonzero_indices)
        min_y, min_x = np.min(nonzero_indices, axis=1)  # 计算最小的 y 和 x 坐标
        max_y, max_x = np.max(nonzero_indices, axis=1)  # 计算最大的 y 和 x 坐标
        if xywh:
            bbox = [int(min_x), int(min_y), int(max_x)-int(min_x), int(max_y)-int(min_y)]
        else:
            bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
        return bbox
    return [0, 0, 0, 0]

def show_mask(img, mask, random_color=False, bbox=None):

    img = cv2.imread(img)

    if random_color:
        color = np.random.random(3)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0
        color = (np.array(cmap(cmap_idx)[:3]) * 255).astype(np.uint8)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, -1) * color.reshape(1, 1, -1)
    img_mask = cv2.addWeighted(img, 1.0, mask_image, 0.6, 1)

    if bbox is not None:
        x, y, w, h = bbox
        img_mask = cv2.rectangle(img_mask, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("show", img_mask)
    cv2.waitKey(100)
    return img_mask

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samsam2",
        action="store_true",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="default",
        help="th, thplus",
    )
    parser.add_argument(
        "--th",
        type=float,
        default=1.0,
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--addon_ths",
        type=str,
        default=None
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_base_plus.pt",
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
        default='results/test/baseline_large',
        help="directory containing input masks (as PNG files) of each video",
    )

    return parser.parse_args()

def main():
    from sam2.build_sam2 import build_sam2_video_predictor

    vis = False
    configs = read_args()
    sam2_checkpoint = configs.checkpoint
    model_cfg = configs.cfg
    dataset_name = configs.dataset
    result_dir = configs.out_dir
    os.makedirs(result_dir, exist_ok=True)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device,
                                           model_selection=configs.method, threshold=configs.th,
                                           addon_ths=configs.addon_ths)

    dataset = LaSOTDataset(dataset_name)
    for i, seq in enumerate(dataset.seqname_list):
        
        results_save_file = os.path.join(result_dir, f"{seq}.txt")
        if os.path.exists(results_save_file):
            print(f"[{i+1}/{len(dataset)}] {seq} - has been tracked before, SKIP!")
            continue
        print(f"[{i+1}/{len(dataset)}] Processing...{seq}")

        inference_state = predictor.init_state(video_path=dataset.get_seq_img_root(i),
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=False,
                                               async_loading_frames=True)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        rect = dataset.get_initial_rect(i)[0]
        x, y, w, h = rect
        box = np.array([x,y,x+w,y+h], dtype=np.float32)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=ann_frame_idx,
                                                        obj_id=ann_obj_id,
                                                        box=box,
                                                        )
        if vis:
            out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            img_paths = dataset.get_seq_frames(i)
            show_mask(img_paths[0], out_mask)

        results = [f"{x},{y},{w},{h}"]
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            # print(out_mask.shape)
            bbox = mask2bbox(out_mask[0], True)
            x, y, w, h = bbox
            if vis:
                show_mask(img_paths[out_frame_idx], out_mask, bbox=bbox)
            if out_frame_idx > 0:
                results.append(f"{x},{y},{w},{h}")
            # print(f"{x},{y},{w},{h}")
        
        results = '\n'.join(results)
        
        with open(results_save_file, 'w') as f: f.write(results)
        predictor.reset_state(inference_state)

if __name__ == "__main__":
    main()