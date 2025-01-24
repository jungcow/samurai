"""
guide
- prompt.txt (has several lines)
    |-- x1, y1, x2, y2, start_idx, end_idx (object1)
    |-- x
"""


import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
import shutil
from tqdm import tqdm

color = [(255, 0, 0)]

def load_txt_from_line(line, prompt_format):
    prompts = {}
    if prompt_format:
        print("Prompt Line: ", repr(line))
        x1, y1, x2, y2, start_frame, end_frame = map(float, line.strip().split(','))
        x1, y1, x2, y2, start_frame, end_frame = int(x1), int(y1), int(x2), int(y2), int(start_frame), int(end_frame)
        prompts[0] = ((x1, y1, x2, y2), (start_frame, end_frame), 0)
    else:
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[0] = ((x, y, x + w, y + h), 0)

    return prompts

def load_txt(gt_path, prompt_format):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        if prompt_format:
            x1, y1, x2, y2, start_frame, end_frame = map(float, line.split(','))
            x1, y1, x2, y2, start_frame, end_frame = int(x1), int(y1), int(x2), int(y2), int(start_frame), int(end_frame)
            prompts[fid] = ((x1, y1, x2, y2), (start_frame, end_frame), 0)
        else:
            x, y, w, h = map(float, line.split(','))
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args, prompt_line, frames, frames_masked):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")

    prompts = load_txt_from_line(prompt_line, args.prompt_format)
    # prompts = load_txt(args.txt_path, args.prompt_format)
    frames_or_path = prepare_frames_or_path(args.video_path)

    bbox, (start_frame, end_frame), track_label = prompts[0]
    if args.save_to_images:
        tmp_frame_folder_path = os.path.join('tmp', f'{start_frame}_{end_frame}')
        os.makedirs(f'{tmp_frame_folder_path}', exist_ok=True)
        for frame in tqdm(frames[start_frame:end_frame], desc="copying images..."):
            shutil.copyfile(frame, os.path.join(tmp_frame_folder_path, os.path.basename(frame)))
        frames_or_path = prepare_frames_or_path(tmp_frame_folder_path)

    frame_rate = 30
    if args.save_to_video:
        if osp.isdir(args.video_path):
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames[start_frame:end_frame]]
            loaded_frames_masked = [cv2.imread(frame_path) for frame_path in frames_masked[start_frame:end_frame]]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(args.video_output_path, f'{start_frame}_{end_frame}.mp4'), fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        print("Found start_frame: ", start_frame)
        print("Found end_frame: ", end_frame)
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_images:
                output_img = loaded_frames_masked[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    output_img[mask] = (0, 0, 0)

                image_path = os.path.join(args.image_output_path, f"{start_frame+frame_idx:06d}.png")
                cv2.imwrite(image_path, output_img)

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

def init_mask(video_path, image_output_path):
    frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
    loaded_frame = cv2.imread(frames[0])
    height, width = loaded_frame.shape[:2]
    for i in tqdm(range(len(frames))):
        output_img = np.full((height,width,3), 255, np.uint8)
        image_path = os.path.join(image_output_path, f"{os.path.basename(frames[i]).replace('jpg', 'png')}")
        cv2.imwrite(image_path, output_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--image_output_path", required=True, help="Path to save the output frames.")
    parser.add_argument("--save_to_images", default=True, help="Save results to a frames.")
    parser.add_argument("--prompt_format", type=str, default="sam", help="sam|samurai")
    parser.add_argument("--init_mask", action="store_true", default=False, help="Wheter initialize the mask folder")
    args = parser.parse_args()

    if args.init_mask:
        os.makedirs(args.image_output_path, exist_ok=True)
        init_mask(args.video_path, args.image_output_path)
    frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
    frames_masked = sorted([osp.join(args.image_output_path, f) for f in os.listdir(args.image_output_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"))])

    txt_path = args.txt_path
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'): continue
            stripped_line = line.strip()
            if not stripped_line: continue
            main(args, stripped_line, frames, frames_masked)
