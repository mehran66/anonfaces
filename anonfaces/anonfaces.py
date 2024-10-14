#!/usr/bin/env python3

import argparse
import json
import mimetypes
from typing import Dict, Tuple
import shutil
import tqdm
import skimage.draw
import numpy as np
import imageio.plugins.ffmpeg
import cv2
import subprocess
import signal
from moviepy.editor import *
from pedalboard import *
from pedalboard.io import AudioFile
from tqdm import tqdm
from io import BytesIO
import ffmpeg
try:
    from centerface import CenterFace  # Import when running as a standalone script
except ImportError:
    from anonfaces.centerface import CenterFace  # Import when used as a library

# Sends a signal to stop ffmpeg
stop_ffmpeg = False

def signal_handler(signum, frame):
    global stop_ffmpeg
    stop_ffmpeg = True
    tqdm.write(f"")
    tqdm.write("Stop signal received, stopping cleanly...")
    tqdm.write(f"")

signal.signal(signal.SIGINT, signal_handler)


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass
    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)

        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()

def get_video_bitrate_ffmpeg(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream and 'bit_rate' in video_stream:
            return int(video_stream['bit_rate'])
    except ffmpeg.Error as e:
        print(f"Error retrieving bitrate: {e.stderr.decode()}")
    return None

def get_video_pix_fmt_ffmpeg(video_path):
    try:
        # Probe video to get metadata
        probe = ffmpeg.probe(video_path)
        # Find the first video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # Check if the pixel format is available in the video stream metadata
        if video_stream and 'pix_fmt' in video_stream:
            return video_stream['pix_fmt']
    except ffmpeg.Error as e:
        print(f'Error retrieving pixel format: {e.stderr.decode()}')
    return None


def has_audio_stream_ffmpeg(video_path):
    try:
        # Probe video to get metadata
        probe = ffmpeg.probe(video_path)
        # Check if any stream is of type 'audio'
        audio_stream = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        return audio_stream
    except ffmpeg.Error as e:
        print(f'Error checking for audio stream: {e.stderr.decode()}')
        return False

def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg=None,
        keep_audio: bool = False,
        copy_acodec: bool = False,
        mosaicsize: int = 20,
        show_ffmpeg_config: bool = False,
        show_ffmpeg_command: bool = False,
):

    # Handle camera input
    if cam:
        cap = cv2.VideoCapture(0)  # Adjust camera index if necessary
    else:
        cap = cv2.VideoCapture(ipath)

    if not cap.isOpened():
        if cam:
            tqdm.write(f'Could not open camera device. Please check your camera connection.')
        else:
            tqdm.write(f'Could not open file {ipath} as a video file with OpenCV. Skipping file...')
        return

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as e:
        tqdm.write(f'Error retrieving video properties: {e}')
        return

    if fps <= 0 or fps > 120:
        fps = 30  # Default to 30 fps if fps is invalid

    # Adjust progress bar
    if nested and not cam:
        bar = tqdm(total=frame_count, position=1, leave=True, dynamic_ncols=True)
    else:
        bar = tqdm(position=1, leave=True, dynamic_ncols=True) if nested else tqdm(dynamic_ncols=True)

    # Prepare ffmpeg_config
    _ffmpeg_config = ffmpeg_config.copy()
    _ffmpeg_config['fps'] = fps
    _ffmpeg_config.setdefault('ffmpeg_log_level', 'panic')

    _ffmpeg_config.setdefault('ffmpeg_params', [])
    _ffmpeg_config['ffmpeg_params'].extend(['-map_metadata', '0', '-map', '0'])

    pix_fmt = get_video_pix_fmt_ffmpeg(ipath)
    common_pix_fmts = ['yuv420p', 'yuvj420p', 'nv12']
    if pix_fmt in common_pix_fmts:
        _ffmpeg_config['ffmpeg_params'].extend(['-pix_fmt', pix_fmt])
    else:
        _ffmpeg_config['ffmpeg_params'].extend(['-pix_fmt', 'yuv420p'])

    input_video_bitrate = get_video_bitrate_ffmpeg(ipath)
    if input_video_bitrate:
        bitrate_kbps = input_video_bitrate // 1000
        _ffmpeg_config['bitrate'] = f'{bitrate_kbps}k'
    else:
        # Set default bitrate if extraction fails
        _ffmpeg_config['bitrate'] = '2000k'

    # Adjust Encoding Settings
    _ffmpeg_config['codec'] = 'libx264'  # Ensure using H.264 codec
    _ffmpeg_config['ffmpeg_params'].extend([
        '-preset', 'slow',
        '-profile:v', 'baseline',  # Match input video's profile
        '-level', '3.1',           # Set level if needed
    ])

    has_audio = has_audio_stream_ffmpeg(ipath)

    # Handle audio settings
    if keep_audio and has_audio:
        _ffmpeg_config['audio_path'] = ipath
        if copy_acodec:
            _ffmpeg_config['audio_codec'] = 'copy'
        else:
            _ffmpeg_config['audio_codec'] = 'aac'
            _ffmpeg_config['audio_bitrate'] = '128k'

    # Prepare writer
    writer = imageio.get_writer(
        opath, format='FFMPEG', mode='I', **_ffmpeg_config
    )

    if show_ffmpeg_config:
        tqdm.write(f'FFMPEG Config: {_ffmpeg_config}')
        tqdm.write("")

    # Construct and display FFmpeg command
    if show_ffmpeg_command:
        ffmpeg_command = (
            f"ffmpeg -y -loglevel {_ffmpeg_config['ffmpeg_log_level']} "
            f"-f rawvideo -vcodec rawvideo -s {frame_width}x{frame_height} -pix_fmt rgb24 -r {_ffmpeg_config['fps']} -i - "
        )
        if 'audio_path' in _ffmpeg_config:
            ffmpeg_command += f"-i {_ffmpeg_config['audio_path']} "
        ffmpeg_command += f"-an -vcodec {_ffmpeg_config['codec']} "
        if 'bitrate' in _ffmpeg_config:
            ffmpeg_command += f"-b:v {_ffmpeg_config['bitrate']} "
        if 'ffmpeg_params' in _ffmpeg_config:
            params = ' '.join(_ffmpeg_config['ffmpeg_params'])
            ffmpeg_command += f"{params} "
        if 'audio_codec' in _ffmpeg_config:
            ffmpeg_command += f"-c:a {_ffmpeg_config['audio_codec']} "
            if 'audio_bitrate' in _ffmpeg_config:
                ffmpeg_command += f"-b:a {_ffmpeg_config['audio_bitrate']} "
        ffmpeg_command += f"{opath}"
        tqdm.write(f"FFMPEG Command: {ffmpeg_command}")
        tqdm.write("")

    # Start processing frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if stop_ffmpeg:
            break

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dets, _ = centerface(frame_rgb, threshold=threshold)

        anonymize_frame(
            dets, frame_rgb, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
            replaceimg=replaceimg, mosaicsize=mosaicsize
        )

        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        if opath is not None:
            writer.append_data(frame_rgb)

        if enable_preview:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame_bgr)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                cv2.destroyAllWindows()
                break

        bar.update(1)

    cap.release()
    if opath is not None:
        writer.close()
    bar.close()



EXTRACTED_AUDIO = "extracted_audio.wav"
DISTORTED_AUDIO = "distorted_audio.wav"


def extract_audio_from_video(v_path: str, a_path: str):
    video = VideoFileClip(v_path)
    video.audio.write_audiofile(a_path)

def distort_audio(audio_input: str, audio_output: str, sample_rate: float = 44100.0):
    with AudioFile(audio_input).resampled_to(sample_rate) as f:
        audio = f.read(f.frames)

    board = Pedalboard([
        Gain(gain_db=5),
        PitchShift(semitones=-2.5),
    ])
    d_audio = board(audio, sample_rate)

    with AudioFile(audio_output, 'w', sample_rate, d_audio.shape[0]) as f:
        f.write(d_audio)

def combine_video_audio(v_path: str, a_path: str, o_path: str):
    vclip = VideoFileClip(v_path)
    aclip = AudioFileClip(a_path)

    vclip.audio = aclip
    vclip.write_videofile(o_path, codec="libx264", logger=None)

def distort_now(ipath, opath):

    # Add "_distorted" to the output file name
    root, ext = os.path.splitext(opath)
    dopath = f"{root}_distorted{ext}"

    # Copy opath to dopath
    shutil.copy(opath, dopath)

    # Extract audio from the original video
    extract_audio_from_video(ipath, EXTRACTED_AUDIO)

    # Distort the extracted audio
    distort_audio(EXTRACTED_AUDIO, DISTORTED_AUDIO)

    # Combine the processed audio with the original video
    combine_video_audio(opath, DISTORTED_AUDIO, dopath)

    # Remove temporary audio files
    os.remove(EXTRACTED_AUDIO)
    os.remove(DISTORTED_AUDIO)


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = imageio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    tqdm.write(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ anonfaces vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ anonfaces cam` (which is a shortcut for `$ anonfaces -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    parser.add_argument(
        '--thresh', '-t', default=0.2, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--distort-audio', '-da', default=False, action='store_true',
        help='Enable audio distortion for the output video (applies pitch shift and gain effects to the audio). This automatically applies --keep-audio but will not work with --copy-acodec due to MoviePy')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--copy-acodec', '-ca', default=False, action='store_true',
        help='Keep audio codec from video source file.')
    parser.add_argument(
        '--show-ffmpeg-config', '-config', action='store_true', default=False,
        help='Shows the FFmpeg config arguments string.'
    )
    parser.add_argument(
        '--show-ffmpeg-command', '-command', action='store_true', default=False,
        help='Shows the FFmpeg constructed command used for processing the video. Helpful for troublshooting.'
    )
    parser.add_argument(
        '--show-both', '-both', action='store_true', default=False,
        help='Shows both --show-ffmpeg-command & --show-ffmpeg-config'
    )
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if args.show_both:
        args.show_ffmpeg_command = True
        args.show_ffmpeg_config = True

    # Automatically enable keep_audio if distort_audio is set
    if args.distort_audio:
        args.keep_audio = True

    if args.keep_audio and args.copy_acodec:
        tqdm.write("")
        tqdm.write("Error: '--keep-audio' and '--copy-acodec' cannot be used together. Please choose one.")
        exit(1)

    if len(args.input) == 0:
        parser.print_help()
        tqdm.write('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    copy_acodec = args.copy_acodec
    ffmpeg_config = args.ffmpeg_config
    show_ffmpeg_config = args.show_ffmpeg_config
    show_ffmpeg_command = args.show_ffmpeg_command
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        if stop_ffmpeg:
            break  # exit the loop immediately if signal is received
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anon{ext}'
        tqdm.write(f'Input:  {ipath}\nOutput: {opath}')
        print()
        if opath is None and not enable_preview:
            tqdm.write('No output file is specified and the preview GUI is disabled. No output will be produced.')

        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                copy_acodec=copy_acodec,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                show_ffmpeg_config=show_ffmpeg_config,
                show_ffmpeg_command=show_ffmpeg_command
            )
            # Check if args.distort_audio is allowed
            if stop_ffmpeg:
                break  # exit the loop immediately if signal is received - second loop
            if args.distort_audio:
                tqdm.write("Distorting audio for the video...")
                distort_now(ipath, opath)
            else:
                tqdm.write("Skipping audio distortion.")
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
            if stop_ffmpeg:
                break  # exit the loop immediately if signal is received - third loop
        elif filetype is None:
            tqdm.write(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            tqdm.write(f'File {ipath} not found. Skipping...')
        else:
            tqdm.write(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()