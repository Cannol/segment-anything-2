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
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

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
        self.D = {} # (frame_index, cls_score)
        self.good_frames = []

    def judge_state(self, output_dict, i):
        if i > 0:
            iou_score = max(output_dict["non_cond_frame_outputs"][i]["best_iou_score"])  # Get mask affinity score
            obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"]  # Get object score
        else:
            iou_score = 1
            obj_score = 1000
        if obj_score > 0 and iou_score > 0.7:
            # if obj_score > 4:
            self.good_frames.append(i)
            label = self.clustered_data["Label"][i]
            frames = self.D.get(label, None)
            if frames is None:
                self.D[label] = [None] * 6
                self.D[label][0] = (obj_score, i)
            else:
                j = 0
                while j < len(frames):
                    tt = frames[j]
                    if tt is not None:
                        score, frame = tt
                        if obj_score > score:
                            break
                    else:
                        break
                    j += 1
                if j < len(frames):
                    frames.insert(j, (obj_score, i))
                    frames.pop()


    def select_frames(self, i, num=7):
        label = self.clustered_data["Label"][i]
        frames = self.D.get(label, None)
        if frames is None:
            self.D[label] = [None]*6
            frames = self.D[label]
        frame_out = set()
        for frame in frames:
            if frame is not None:
                frame_out.add(frame[1])
        # if len(frame_out) >= num:
        #     frame_out = list(frame_out)
        #     frame_out.sort()
        #     return frame_out, label
        for frame in self.good_frames[::-1]:
            frame_out.add(frame)
            if len(frame_out) >= num:
                break
        frame_out = list(frame_out)
        frame_out.sort()
        while len(frame_out) < num:
            frame_out.insert(1,None)
        return frame_out, label

    def finetune_samples(self, frames, flag):
        if frames[0] != 0:
            frames[0] = 0
        return frames

    @torch.inference_mode()
    def calculate_all_frames(self, inference_state):
        self.clustered_data = None
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
        all_label_types = set(data["Label"])
        print(f"clustering classes: {len(all_label_types)}")
        self.clustered_data = data

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, *args, **kwargs):
        self.calculate_all_frames(inference_state)
        return super().propagate_in_video(inference_state, *args, **kwargs)

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            self.judge_state(output_dict, frame_idx-1)
            proposed_frames, flag = self.select_frames(frame_idx)
            proposed_frames = self.finetune_samples(proposed_frames, flag)
            # print(f"type: {flag} -- {proposed_frames}")
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            t_pos_and_prevs = []
            for t_pos, proposed_frame in enumerate(proposed_frames):
                if proposed_frame is None:
                    t_pos_and_prevs.append((t_pos, None))
                else:
                    out = output_dict["cond_frame_outputs"].get(proposed_frame, None)
                    if out is None:
                        out = output_dict["non_cond_frame_outputs"].get(proposed_frame, None)
                    t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in output_dict["cond_frame_outputs"].items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = output_dict["cond_frame_outputs"]

                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, None
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # print(pos_list)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            # self.good_frames.append(0)
            # self.select_frames(0)
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem