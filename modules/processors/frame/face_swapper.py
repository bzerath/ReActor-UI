from typing import Any, List
import cv2
import insightface
from insightface.model_zoo.inswapper import INSwapper
import numpy
import threading

import modules.variables.values
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, get_best_one_face, get_face_analyser, extract_all_faces
from modules.variables.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
SHORTNAME = "FACE-SWAPPER"
NAME = f'REACTOR.{SHORTNAME}'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(modules.variables.values.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(modules.variables.values.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.variables.values.target_path) and not is_video(modules.variables.values.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> INSwapper:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.variables.values.execution_providers)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame, subject_embedding: Face) -> Frame:
    """
    :param source_face: get_one_face(cv2.imread(source_path))
    :param temp_frame: cv2.imread(temp_frame_path)
    :param subject_frame: cv2.imread(subject_frame_path)
    :return:
    """
    if modules.variables.values.face_option == modules.variables.values.faces_all:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    elif modules.variables.values.face_option == modules.variables.values.faces_best_one:
        target_face = get_best_one_face(temp_frame, subject_embedding)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        pass
    return temp_frame


def debug_frame(source_frame: Face, temp_frame: Frame, subject_embedding: Face) -> Frame:
    faces_from_frame = get_face_analyser().get(temp_frame)
    for face in faces_from_frame:
        match_score = numpy.linalg.norm(subject_embedding - face.embedding)
        bbox = face.bbox[:4].astype(int)
        cv2.rectangle(temp_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        text = f'Score: {match_score:.2f}'
        cv2.putText(temp_frame,
                    text,
                    (max(bbox[0], 0), max(bbox[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return temp_frame

def process_frames(source_path: str,
                   temp_frame_paths: List[str],
                   subject_path: str,
                   progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path))

    subject_face = get_one_face(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face.embedding
    else:
        raise Exception("Subject face does not contain face...")

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame(source_face, temp_frame, subject_embedding)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def debug_frames(source_path: str,
                 temp_frame_paths: List[str],
                 subject_path: str,
                 progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path))

    subject_face = get_one_face(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face.embedding
    else:
        raise Exception("Subject face does not contain face...")

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = debug_frame(source_face, temp_frame, subject_embedding)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, subject_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    subject_face = get_face_analyser().get(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face[0].embedding
    else:
        raise Exception("Subject face does not contain face...")
    result = process_frame(source_face, target_frame, subject_embedding)
    cv2.imwrite(output_path, result)


def debug_image(source_path: str, target_path: str, subject_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    subject_face = get_face_analyser().get(cv2.imread(subject_path))
    if subject_face:
        subject_embedding = subject_face[0].embedding
    else:
        raise Exception("Subject face does not contain face...")
    result = debug_frame(source_face, target_frame, subject_embedding)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], subject_path: str) -> None:
    modules.processors.frame.core.process_video(source_path,
                                                temp_frame_paths,
                                                process_frames,
                                                subject_path)


def debug_video(source_path: str, temp_frame_paths: List[str], subject_path: str) -> None:
    modules.processors.frame.core.debug_video(source_path, temp_frame_paths, debug_frames, subject_path)
