import os
import sys

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Literal
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
import modules.utilities as utilities
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_unsound_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, \
    normalize_output_path

if 'ROCMExecutionProvider' in modules.variables.values.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()

    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')

    args = program.parse_args()

    modules.variables.values.execution_providers = decode_execution_providers(args.execution_provider)
    modules.variables.values.execution_threads = suggest_execution_threads()
    modules.variables.values.max_memory = suggest_max_memory()


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower()
            for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider
            in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
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


def process(process: Literal["process", "debug"]) -> None:
    for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
        if not frame_processor.pre_start():
            return
    if has_image_extension(modules.variables.values.target_path):
        if not modules.variables.values.nsfw:
            from modules.predicter import predict_image
            if predict_image(modules.variables.values.target_path):
                destroy()
        shutil.copy2(modules.variables.values.target_path, modules.variables.values.output_path)
        for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            method = getattr(frame_processor, process + "_image")
            method(modules.variables.values.source_path,
                   modules.variables.values.output_path,
                   modules.variables.values.subject_path,
                   modules.variables.values.output_path)
            release_resources()
        if is_image(modules.variables.values.output_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if not modules.variables.values.nsfw:
        from modules.predicter import predict_video
        if predict_video(modules.variables.values.target_path):
            destroy()

    if modules.variables.values.decompose_video:
        update_status('Creating temp resources...')
        create_temp(modules.variables.values.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.variables.values.target_path)
    else:
        update_status('Keeping frames existing.')

    temp_frame_paths = get_temp_frame_paths(modules.variables.values.target_path)
    for frame_processor in get_frame_processors_modules(modules.variables.values.frame_processors):
        update_status('Progressing... source_path={}'.format(modules.variables.values.source_path),
                      frame_processor.NAME)
        method = getattr(frame_processor, process + "_video")
        method(source_path=modules.variables.values.source_path,
               temp_frame_paths=temp_frame_paths,
               subject_path=modules.variables.values.subject_path)
        release_resources()

    update_status(f'Creating video...')

    if modules.variables.values.recompose_video:
        utilities.create_video(target_path=modules.variables.values.target_path,
                               output_path=modules.variables.values.output_path)
    else:
        update_status("Not recomposing video.")
    # clean and validate
    clean_temp(modules.variables.values.target_path)
    if is_video(modules.variables.values.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def start() -> None:
    process("process")


def debug() -> None:
    process("debug")


def destroy() -> None:
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        print("pre_check KO")
        return
    limit_resources()
    window = ui.App(start=start, debug=debug)
    window.mainloop()
