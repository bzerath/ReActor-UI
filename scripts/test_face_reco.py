import face_recognition
import os
import numpy


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


ANNECLAIRE = [
    resolve_relative_path("../face_reco/ref/Anne-Claire01.JPG"),
    resolve_relative_path("../face_reco/ref/Anne-Claire02.JPG"),
    # resolve_relative_path("../face_reco/ref/Anne-Claire03.JPG"),
    resolve_relative_path("../face_reco/ref/Anne-Claire04.JPG"),
    resolve_relative_path("../face_reco/ref/Anne-Claire05.JPG"),
]

ALICE = [
    resolve_relative_path("../face_reco/ref/Alice01.jpeg"),
    resolve_relative_path("../face_reco/ref/Alice02.JPG"),
    resolve_relative_path("../face_reco/ref/Alice03.JPG"),
    resolve_relative_path("../face_reco/ref/Alice04.JPG"),
    resolve_relative_path("../face_reco/ref/Alice05.jpg"),
]


def get_distance(reference_path: str,
                 test_path: str) -> float:
    reference_image = face_recognition.load_image_file(reference_path)
    reference_encoding = face_recognition.face_encodings(reference_image)
    if reference_encoding:
        test_image = face_recognition.load_image_file(test_path)
        test_encoding = face_recognition.face_encodings(test_image)
        return face_recognition.face_distance([reference_encoding], test_encoding)


def isGoodFace(references_encodings: list[numpy.ndarray],
               to_test_encoding: numpy.ndarray,
               tolerance: float = 0.6) -> bool:
    results = list(face_recognition.face_distance(references_encodings, to_test_encoding))
    return sum(results)/len(results) <= tolerance


if __name__ == '__main__':

    references_encodings = []

    for image_path in ALICE:
        reference_image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(reference_image)
        if encoding:
            references_encodings.append(encoding[0])

    unknown_image = face_recognition.load_image_file(resolve_relative_path("../face_reco/test/IMG-20150829-WA0014.jpg"))

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        print(isGoodFace(references_encodings=references_encodings,
                         to_test_encoding=unknown_encoding,
                         tolerance=0.5))
