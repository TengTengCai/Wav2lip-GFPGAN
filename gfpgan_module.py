import glob
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from basicsr import imwrite
from cv2 import cv2
from gfpgan import GFPGANer

from tqdm import tqdm

logger = logging.getLogger(__name__)


def video_to_frame(input_video_path, un_processed_frames_folder_path):
    if not os.path.exists(un_processed_frames_folder_path):
        os.makedirs(un_processed_frames_folder_path)

    vidcap = cv2.VideoCapture(input_video_path)
    numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    logger.info(f"FPS: {fps}, Frames: {numberOfFrames}")

    for frameNumber in tqdm(range(numberOfFrames)):
        _, image = vidcap.read()
        cv2.imwrite(os.path.join(un_processed_frames_folder_path, str(frameNumber).zfill(4) + '.jpg'), image)


def frame_to_video(restored_frames_path, output_path, fps=25, size=None):
    processedVideoOutputPath = output_path

    dir_list = os.listdir(restored_frames_path)
    dir_list.sort()

    batch = 0
    batchSize = 300

    if size is None:
        size = (1280, 720)
    for i in tqdm(range(0, len(dir_list), batchSize)):
        img_array = []
        s, e = i, i + batchSize
        logger.info(f"processing {s} {e}")

        for filename in tqdm(dir_list[s:e]):
            filename = restored_frames_path + filename
            img = cv2.imread(filename)
            if img is None:
                continue
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(processedVideoOutputPath + '/batch_' + str(batch).zfill(4) + '.avi',
                              cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        batch = batch + 1

        for j in range(len(img_array)):
            out.write(img_array[j])
        out.release()

    concatTextFilePath = output_path + "/concat.txt"
    concatTextFile = open(concatTextFilePath, "w")
    for ips in range(batch):
        concatTextFile.write("file batch_" + str(ips).zfill(4) + ".avi\n")
    concatTextFile.close()
    return concatTextFilePath


def process_image(img_path, output_path, restorer, ):
    # read image
    img_name = os.path.basename(img_path)
    logger.info(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5)

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # save cropped face
        save_crop_path = os.path.join(output_path, 'cropped_faces', f'{basename}_{idx:02d}.png')
        imwrite(cropped_face, save_crop_path)
        # save restored face
        save_face_name = f'{basename}_{idx:02d}.png'
        save_restore_path = os.path.join(output_path, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)
        # save comparison image
        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        imwrite(cmp_img, os.path.join(output_path, 'cmp', f'{basename}_{idx:02d}.png'))

    # save restored img
    if restored_img is not None:
        extension = ext[1:]
        save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}.{extension}')
        imwrite(restored_img, save_restore_path)


def inference(input_path, output_path, bg_upsampler_type='realesrgan', bg_tile=400, version='1.3', upscale=2,
              max_workers=3):
    """
    Inference demo for GFPGAN (for users).
    """
    # ------------------------ input & output ------------------------
    if input_path.endswith('/'):
        input_path = input_path[:-1]
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        img_list = sorted(glob.glob(os.path.join(input_path, '*')))

    os.makedirs(output_path, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler_type == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='gfpgan/weights/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img_path = {
            executor.submit(process_image, img_path, output_path, restorer): img_path for img_path in img_list
        }
        for future in tqdm(as_completed(future_to_img_path)):
            future.result()
            img_path = future_to_img_path[future]
            logger.info(f"Finish {img_path}")

    logger.info(f'Results are in the [{output_path}] folder.')


def start(input_video, output_path='results/gfpgan', max_workers=3, fps=25):
    un_processed_frames_folder_path = 'results/wav2lip/frames'
    os.makedirs(un_processed_frames_folder_path, exist_ok=True)
    restored_frames_path = os.path.join(output_path, '/frames')
    video_output_path = os.path.join(output_path, '/video')
    concated_video_output_path = os.path.join(output_path, '/concated/concated_output.avi')
    video_to_frame(input_video, un_processed_frames_folder_path)
    inference(un_processed_frames_folder_path, restored_frames_path, max_workers=max_workers)
    concatTextFilePath = frame_to_video(restored_frames_path, video_output_path, fps=fps)
    command = f'ffmpeg -y -f concat -i {concatTextFilePath} -c copy {concated_video_output_path}'
    subprocess.call(command, shell=True)
    return concated_video_output_path


if __name__ == '__main__':
    start('results/result_voice.mp4')
