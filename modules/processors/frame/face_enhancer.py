from typing import Any, List
import cv2
import threading
import gfpgan

import modules.variables.values
import modules.processors.frame.core
from modules.face_analyser import get_one_face, get_face_analyser, extract_best_one_face, extract_all_faces
from modules.variables.typing import Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
SHORTNAME = "FACE-ENHANCER"
NAME = f'REACTOR.{SHORTNAME}'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(modules.variables.values.target_path) and not is_video(modules.variables.values.target_path):
        print('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
    return FACE_ENHANCER


def enhance_face(temp_frame: Frame, ref_embedding: Frame) -> Frame:
    if modules.variables.values.enhancer_option == modules.variables.values.enhancer_faces_only:
        for (top, left, bottom, right), face in extract_all_faces(temp_frame):
            try:
                with THREAD_SEMAPHORE:
                    _, _, face = get_face_enhancer().enhance(
                        face,
                        paste_back=True
                    )
                face = cv2.resize(face, (right - left, bottom - top))
                temp_frame[top:bottom, left:right] = face
            except Exception as e:
                pass
    elif modules.variables.values.enhancer_option == modules.variables.values.enhancer_best_face_only:
        face = extract_best_one_face(temp_frame, ref_embedding)
        if face:
            top, left, bottom, right, face_frame = extract_best_one_face(temp_frame, ref_embedding)
            try:
                with THREAD_SEMAPHORE:
                    _, _, face_frame = get_face_enhancer().enhance(
                        face_frame,
                        paste_back=True
                    )
                face_frame = cv2.resize(face_frame, (right - left, bottom - top))
                temp_frame[top:bottom, left:right] = face_frame
            except Exception as e:
                pass
    elif modules.variables.values.enhancer_option == modules.variables.values.enhancer_all:
        try:
            with THREAD_SEMAPHORE:
                _, _, temp_frame = get_face_enhancer().enhance(
                    temp_frame,
                    paste_back=True
                )
        except Exception as e:
            pass
    else:
        pass

    return temp_frame


def process_frame(temp_frame: Frame, ref_embedding: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, ref_embedding)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], subject_path: str, progress: Any = None) -> None:
    subject_face = get_face_analyser().get(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face[0].embedding
    else:
        raise Exception("Subject face does not contain face...")
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(temp_frame, subject_embedding)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, subject_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    subject_face = get_face_analyser().get(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face[0].embedding
    else:
        raise Exception("Subject face does not contain face...")
    result = process_frame(target_frame, subject_embedding)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], subject_path: str) -> None:
    modules.processors.frame.core.process_video(source_path=None,
                                                temp_frame_paths=temp_frame_paths,
                                                process_frames=process_frames,
                                                subject_path=subject_path)
