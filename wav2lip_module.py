# This is a sample Python script.
import os
import platform
import subprocess
import logging

import cv2
import numpy as np
import torch
from tqdm import tqdm

from Wav2Lip import audio, face_detection
from Wav2Lip.models import Wav2Lip

logger = logging.getLogger(__name__)


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, batch_size=16, pads=None, nosmooth=False):
    if pads is None:
        pads = (0, 10, 0, 0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = face_detection.FaceAlignment(1, flip_input=False, device=device)

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels, box=None, static=False, img_size=96, wav2lip_batch_size=128):
    if box is None:
        box = (-1, -1, -1, -1)
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using {} for inference.'.format(device))
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def start(face_path, audio_path, outfile='results/result_voice.mp4', checkpoint_path='Wav2Lip/weights/wav2lip.pth',
          fps=25, resize_factor=1, rotate=False, crop=None, wav2lip_batch_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if crop is None:
        crop = (0, -1, 0, -1)
    mel_step_size = 16
    if not os.path.isfile(face_path):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face_path)]
    else:
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        logger.info('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    logger.info("Number of frames available for inference: " + str(len(full_frames)))

    if not audio_path.endswith('.wav'):
        logger.info('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    logger.info(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    logger.info("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)
    model = load_model(checkpoint_path)
    logger.info("Model loaded...")

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    return outfile


def main():
    start('test/test.jpg', 'test/ttsmaker-file-2023-2-21-15-47-32.mp3')


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Using {} for inference.'.format(device))
    main()
