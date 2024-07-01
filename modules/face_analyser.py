from typing import Any
import insightface
import numpy

import modules.variables.values
from modules.typing import Face, Frame

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.variables.values.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def extract_best_one_face(source_frame: Frame, ref_embedding: Frame) -> Frame:
    best_one_face = get_best_one_face(source_frame, ref_embedding)
    if not best_one_face:
        print("No face detected in source_frame")
        return
    bbox = best_one_face.bbox[:4].astype(int)  # Les coordonnées de la boîte englobante
    top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]

    return top, left, bottom, right, source_frame[top:bottom, left:right]


def extract_all_faces(source_frame: Frame) -> list[tuple[tuple, Frame]]:
    output = []
    faces = get_face_analyser().get(source_frame)
    for face in faces:
        bbox = face.bbox[:4].astype(int)  # Les coordonnées de la boîte englobante
        top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
        output.append(((top, left, bottom, right), source_frame[top:bottom, left:right]))

    return output


def get_best_one_face(source_frame: Frame, ref_embedding: Frame) -> Face:
    """
    Return the face from source_frame with best ressemblance to ref_frame.
    :param source_frame: photo to analyze
    :param ref_frame: face to search
    :return:
    """
    faces_from_frame = get_face_analyser().get(source_frame)
    if faces_from_frame:
        if len(faces_from_frame) == 1:
            return faces_from_frame[0]
        else:
            best_score = modules.variables.values.distance_score
            selected_face = None
            for face in faces_from_frame:
                test_embedding = face.embedding
                match_score = numpy.linalg.norm(ref_embedding - test_embedding)
                if match_score < best_score:
                    selected_face = face
                    best_score = match_score

            return selected_face


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
