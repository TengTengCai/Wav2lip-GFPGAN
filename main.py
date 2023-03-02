import argparse
import logging
import subprocess

import gfpgan_module
import wav2lip_module

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)
args = parser.parse_args()


def main():
    outfile = wav2lip_module.start(args.face, args.audio, fps=args.fps)
    concated_video_output_path = gfpgan_module.start(outfile, fps=args.fps)
    final_processed_ouput_video = 'results/final_with_audio.avi'
    command = f'ffmpeg -y -i {concated_video_output_path} -i {args.audio} ' \
              f'-map 0 -map 1:a -c:v copy {final_processed_ouput_video}'
    subprocess.call(command, shell=True)
    pass


if __name__ == '__main__':
    main()
