import datetime
import glob
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from typing import List, Dict

import numpy as np
from basicsr import imwrite
from cv2 import cv2

from gfpgan import GFPGANer

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProcessImageWorker(Process):
    def __init__(self, index: int, in_queue: Queue, out_queue: Queue, upscale=1, version="1.3"):
        self.name = f"worker-{index}"
        super().__init__(name=self.name)

        self.in_queue = in_queue
        self.out_queue = out_queue
        self.upscale = upscale

        if version == '1':
            self.arch = 'original'
            self.channel_multiplier = 1
            self.model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif version == '1.2':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif version == '1.3':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif version == '1.4':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif version == 'RestoreFormer':
            self.arch = 'RestoreFormer'
            self.channel_multiplier = 2
            self.model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {version}.')

        # determine model paths
        self.model_path = os.path.join('experiments/pretrained_models', self.model_name + '.pth')
        if not os.path.isfile(self.model_path):
            self.model_path = os.path.join('gfpgan/weights', self.model_name + '.pth')
        if not os.path.isfile(self.model_path):
            # download pre-trained models from url
            self.model_path = url

    def run(self) -> None:
        restorer = GFPGANer(
            model_path=self.model_path,
            upscale=self.upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=None)
        with tqdm(total=self.in_queue.qsize(), desc=f"{self.name}-处理视频") as pbar:
            for _ in range(self.in_queue.qsize()):
                image = self.in_queue.get()
                pbar.update(1)
                _, _, restored_img = restorer.enhance(
                    image,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                    weight=0.5)
                if restored_img is not None:
                    self.out_queue.put(restored_img)


class GFPGANController(object):
    def __init__(self, worker_num=2):
        self.worker_num = worker_num
        self.in_queue_list = [Queue() for _ in range(self.worker_num)]
        self.out_queue_list = [Queue() for _ in range(self.worker_num)]
        self.fps = 25
        self.width, self.height = (1280, 720)
        pass

    def start_worker(self):
        pass

    def read_local_file(self, input_video_path):
        video_cap = cv2.VideoCapture(input_video_path)
        number_of_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = video_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"FPS: {self.fps}, Frames: {number_of_frames}")
        with tqdm(total=number_of_frames, desc="读取视频帧") as pbar:
            index = 0
            for item in np.array_split(np.array(range(number_of_frames)), len(self.in_queue_list)):
                this_queue = self.in_queue_list[index]
                for _ in item:
                    ret, image = video_cap.read()
                    pbar.update(1)
                    if ret:
                        this_queue.put(image)
                index += 1

    def write_local_file(self, output_video_path):
        date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        size = (self.width, self.height)
        out = cv2.VideoWriter(os.path.join(output_video_path, f'result_{date_str}.avi'),
                              cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
        total = 0
        for out_queue in self.out_queue_list:
            total += out_queue.qsize()
        with tqdm(total=total, desc=f"输出视频帧") as pbar:
            for out_queue in self.out_queue_list:
                for _ in range(out_queue.qsize()):
                    image = out_queue.get()
                    out.write(image)
                    pbar.update(1)
        out.release()

    def handle_local_file(self, input_video_path, output_video_path):
        self.read_local_file(input_video_path)

        for i in range(self.worker_num):
            p = ProcessImageWorker(i, self.in_queue_list[i], self.out_queue_list[i])
            p.start()
            p.join()

        self.write_local_file(output_video_path)


if __name__ == '__main__':
    c = GFPGANController()
    c.handle_local_file("results/result_voice.mp4", "results/")
    time.sleep(600)
    pass
