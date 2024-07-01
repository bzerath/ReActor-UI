import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import modules.variables.values
import modules.variables.metadata
import modules.ui.ui_new as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.variables.values.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.variables.metadata.name} {modules.variables.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.variables.values.source_path = args.source_path
    modules.variables.values.target_path = args.target_path
    modules.variables.values.output_path = normalize_output_path(modules.variables.values.source_path, modules.variables.values.target_path, args.output_path)
    modules.variables.values.frame_processors = args.frame_processor
    modules.variables.values.headless = args.source_path or args.target_path or args.output_path
    modules.variables.values.keep_fps = args.keep_fps
    modules.variables.values.keep_audio = args.keep_audio
    modules.variables.values.keep_frames = args.keep_frames
    modules.variables.values.many_faces = args.many_faces
    modules.variables.values.video_encoder = args.video_encoder
    modules.variables.values.video_quality = args.video_quality
    modules.variables.values.max_memory = args.max_memory
    modules.variables.values.execution_providers = decode_execution_providers(args.execution_provider)
    modules.variables.values.execution_threads = args.execution_threads

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.variables.values.fp_ui['face_enhancer'] = True
    else:
        modules.variables.values.fp_ui['face_enhancer'] = False
    
    modules.variables.values.nsfw = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.variables.values.source_path = args.source_path_deprecated
        modules.variables.values.output_path = normalize_output_path(args.source_path_deprecated, modules.variables.values.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.variables.values.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.variables.values.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.variables.values.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.variables.values.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.variables.values.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.variables.values.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.variables.values.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.variables.values.max_memory:
        memory = modules.variables.values.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.variables.values.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.variables.values.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    """
    Verify that python >= 3.10
    Verify that ffmpeg is installed
    """
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'REACTOR.CORE') -> None:
    print(f'[{scope}] {message}')
    # if not modules.variables.values.headless:
    #     ui.update_status(message)


def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(modules.variables.values.target_path):
        if modules.variables.values.nsfw == False:
            from modules.predicter import predict_image
            if predict_image(modules.variables.values.target_path):
                destroy()
        shutil.copy2(modules.variables.values.target_path, modules.variables.values.output_path)
        for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.variables.values.source_path,
                                          modules.variables.values.output_path,
                                          modules.variables.values.subject_path,
                                          modules.variables.values.output_path)
            release_resources()
        if is_image(modules.variables.values.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if modules.variables.values.nsfw == False:
        from modules.predicter import predict_video
        if predict_video(modules.variables.values.target_path):
            destroy()
    update_status('Creating temp resources...')
    create_temp(modules.variables.values.target_path)
    update_status('Extracting frames...')
    extract_frames(modules.variables.values.target_path)
    temp_frame_paths = get_temp_frame_paths(modules.variables.values.target_path)
    for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
        update_status('Progressing... source_path={}'.format(modules.variables.values.source_path),
                      frame_processor.NAME)
        frame_processor.process_video(source_path=modules.variables.values.source_path,
                                      temp_frame_paths=temp_frame_paths,
                                      subject_path=modules.variables.values.subject_path)
        release_resources()
    # handles fps
    update_status('Detecting fps...')
    fps = detect_fps(modules.variables.values.target_path)
    update_status(f'Creating video with {fps} fps...')
    create_video(modules.variables.values.target_path, fps)
    # handle audio
    if modules.variables.values.keep_audio:
        if modules.variables.values.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.variables.values.target_path, modules.variables.values.output_path)
    else:
        move_temp(modules.variables.values.target_path, modules.variables.values.output_path)
    # clean and validate
    clean_temp(modules.variables.values.target_path)
    if is_video(modules.variables.values.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if modules.variables.values.target_path:
        clean_temp(modules.variables.values.target_path)
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        print("pre_check KO")
        return
    for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
        if not frame_processor.pre_check():
            print("frame_processor {} pre_check KO".format(frame_processor.NAME))
            return
    limit_resources()
    if modules.variables.values.headless:
        start()
    else:
        window = ui.App(start=start)  # ui.init(start, destroy)
        window.mainloop()
