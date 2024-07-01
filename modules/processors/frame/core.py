import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.variables.values

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise AttributeError("{} missing from module {}".format(method_name,
                                                                        frame_processor_module))
                # sys.exit()
    except ImportError as e:
        raise e
        # sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.variables.values.fp_ui.items():
        if state and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.variables.values.frame_processors.append(frame_processor)
        if not state:
            frame_processor_module = load_frame_processor_module(frame_processor)
            try:
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.variables.values.frame_processors.remove(frame_processor)
            except:
                pass


def multi_process_frame(source_path: str,
                        temp_frame_paths: List[str],
                        process_frames: Callable[[str, List[str], str, Any], None],
                        subject_path: str,
                        progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=modules.variables.values.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], subject_path, progress)
            futures.append(future)
        for future in futures:
            future.result()


def process_video(source_path: str,
                  temp_frame_paths: list[str],
                  process_frames: Callable[[str, List[str], str, Any], None],
                  subject_path: str) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(temp_frame_paths)
    with tqdm(total=total,
              desc='Processing',
              unit='frame',
              dynamic_ncols=True,
              bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': modules.variables.values.execution_providers,
                              'execution_threads': modules.variables.values.execution_threads,
                              'max_memory': modules.variables.values.max_memory})
        multi_process_frame(source_path,
                            temp_frame_paths,
                            process_frames,
                            subject_path,
                            progress)
