from abc import ABC, abstractmethod
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


class FaceModifier(ABC):
    ENGINE: INSwapper
    THREAD_LOCK = threading.Lock()

    @abstractmethod
    def get_engine(self) -> INSwapper:
        """ Set self.ENGINE by calling model"""
        ...

    @property
    @abstractmethod
    def shortname(self) -> str:
        ...

    @shortname.setter
    @abstractmethod
    def shortname(self, name: str) -> str:
        ...

    @property
    def name(self) -> str:
        return f'REACTOR.{self.shortname}'

    @abstractmethod
    def pre_check(self) -> bool:
        ...

    @abstractmethod
    def pre_start(self) -> bool:
        ...

    @abstractmethod
    def process_engine(self,
                       source_face: Face,
                       target_face: Face,
                       temp_frame: Frame) -> Frame:
        """
        # TODO - ambition = doit être aussi simple que face_swapper.swap_face. Vraiment juste l'appel à la fonction maîtresse
        ex:
            with THREAD_SEMAPHORE:
                _, _, temp_frame = get_face_enhancer().enhance(
                    temp_frame,
                    paste_back=True
                )
            OU BIEN
            face = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

        à voir l'utilité du semaphore

        :param source_face:
        :param target_face:
        :param temp_frame:
        :return:
        """
        ...

    def process_frame(self,
                      source_face: Face,
                      temp_frame: Frame,
                      subject_embedding: Face,
                      engine_option: str) -> Frame:
        # TODO - ambition = doit contenir toute l'intelligence sur le visage à modifier.
        if modules.variables.values.face_option == modules.variables.values.option_all:
            ...
        elif modules.variables.values.face_option == modules.variables.values.option_all_faces:
            ...
        elif modules.variables.values.face_option == modules.variables.values.option_best_one:
            ...
        else:
            pass
        return temp_frame

    def process_frames(self,
                       source_path: str,
                       temp_frame_paths: List[str],
                       subject_path: str,
                       engine_option: str,
                       progress: Any = None) -> None:
        source_face = get_one_face(cv2.imread(source_path))
        if source_face:
            source_embedding = source_face.embedding
        else:
            raise Exception("Source face does not contain face...")

        subject_face = get_one_face(cv2.imread(subject_path))
        if subject_face:
            subject_embedding = subject_face.embedding
        else:
            raise Exception("Subject face does not contain face...")

        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = self.process_frame(source_face=source_face,
                                            temp_frame=temp_frame,
                                            subject_embedding=subject_embedding,
                                            engine_option=engine_option)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)
        pass

    def process_image(self,
                      source_path: str,
                      target_path: str,
                      subject_path: str,
                      output_path: str,
                      engine_option: str) -> None:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        subject_face = get_face_analyser().get(cv2.imread(subject_path))
        if subject_face:
            subject_embedding = subject_face[0].embedding
        else:
            raise Exception("Subject face does not contain face...")
        result = self.process_frame(source_face=source_face,
                                    temp_frame=target_frame,
                                    subject_embedding=subject_embedding,
                                    engine_option=engine_option)
        cv2.imwrite(output_path, result)

    def process_video(self,
                      source_path: str,
                      temp_frame_paths: List[str],
                      subject_path: str) -> None:
        modules.processors.frame.core.process_video(source_path,
                                                    temp_frame_paths,
                                                    self.process_frames,
                                                    subject_path)



