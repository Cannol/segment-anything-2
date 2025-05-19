from collections import OrderedDict

import torch
# import torch.nn.functional as F
#
# from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from sam2.sam2_video_predictor import NO_OBJ_SCORE, SAM2VideoPredictor
# from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames

class SAMSAM2VideoPredictor(SAM2VideoPredictor):

    def __init__(
        self,
        DBSCAN_eps=0.03,
        DBSCAN_min_samples=25,
        pooling_size=256,
        pca_n_components=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cluster = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples)
        self.feature_pooling = torch.nn.AvgPool1d(pooling_size)
        self.pca = PCA(n_components=pca_n_components)
        self.clustered_data = None

    @torch.inference_mode()
    def calculate_all_frames(self, inference_state):
        vis_data = []
        device_inside = inference_state["device"]
        for frame_index in range(inference_state["num_frames"]):
            image = inference_state["images"][frame_index].to(device_inside).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"] = {frame_index: (image, backbone_out)}
            visf = backbone_out["vision_features"]
            # b, c, h, w -> b, h, w, c
            visf = visf.permute(0, 2, 3, 1)
            visf = visf.reshape(1, -1, 256)
            visf = self.feature_pooling(visf)
            visf = visf.detach().cpu().numpy().squeeze().reshape(visf.shape[1], -1)
            vis_data.append(visf)
            if frame_index > 0 and frame_index % 100 == 0:
                print(f"[clustering progress: {frame_index + 1}/{inference_state['num_frames']}]")

        data = np.stack(vis_data, axis=1).squeeze()
        data_pca = data.transpose()
        data = self.pca.fit_transform(data_pca)
        data = pd.DataFrame(data, columns=['X', 'Y'])
        data['Label'] = self.cluster.fit_predict(data)
        self.clustered_data = data

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, *args, **kwargs):
        self.calculate_all_frames(inference_state)
        return super().propagate_in_video(inference_state, *args, **kwargs)