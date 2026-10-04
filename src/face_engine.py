from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

YUNET_MODEL = (
    BASE_DIR
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = (
    BASE_DIR
    / "models"
    / "face_recognition_sface_2021dec.onnx"
)


class FaceEngine:

    def __init__(self):

        if not YUNET_MODEL.exists():
            raise FileNotFoundError(
                f"YuNet model not found:\n{YUNET_MODEL}"
            )

        if not SFACE_MODEL.exists():
            raise FileNotFoundError(
                f"SFace model not found:\n{SFACE_MODEL}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",
            (320, 320),
            0.85,
            0.3,
            5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(SFACE_MODEL),
            ""
        )

    def detect_faces(self, image):

        height, width = image.shape[:2]

        self.detector.setInputSize(
            (width, height)
        )

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        return faces

    def get_face_feature(self, image, face):

        aligned_face = self.recognizer.alignCrop(
            image,
            face
        )

        feature = self.recognizer.feature(
            aligned_face
        )

        return feature

    def compare_features(
        self,
        reference_feature,
        query_feature
    ):

        score = self.recognizer.match(
            reference_feature,
            query_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        return float(score)

    def get_reference_feature(self, image):

        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None

        if len(faces) > 1:
            raise ValueError(
                "Reference image must contain exactly one face."
            )

        return self.get_face_feature(
            image,
            faces[0]
        )

    def find_matching_face(
        self,
        classroom_image,
        reference_feature,
        threshold=0.363
    ):

        faces = self.detect_faces(classroom_image)

        results = []

        for face in faces:

            query_feature = self.get_face_feature(
                classroom_image,
                face
            )

            score = self.compare_features(
                reference_feature,
                query_feature
            )

            results.append(
                {
                    "face": face,
                    "score": score,
                    "match": score >= threshold
                }
            )

        return results

    def recognize_students(
        self,
        classroom_image,
        registered_students,
        threshold=0.363
    ):

        detected_faces = self.detect_faces(
            classroom_image
        )

        results = []

        for face_index, face in enumerate(
            detected_faces
        ):

            query_feature = self.get_face_feature(
                classroom_image,
                face
            )

            best_student = None
            best_score = -1.0

            for student in registered_students:

                feature_path = Path(
                    student["feature"]
                )

                if not feature_path.exists():
                    continue

                reference_feature = np.load(
                    feature_path
                )

                score = self.compare_features(
                    reference_feature,
                    query_feature
                )

                if score > best_score:
                    best_score = score
                    best_student = student

            if (
                best_student is not None
                and best_score >= threshold
            ):

                result = {
                    "student_id": best_student["student_id"],
                    "name": best_student["name"],
                    "college": best_student["college"],
                    "score": float(best_score),
                    "match": True,
                    "face_index": face_index,
                    "face": face
                }

            else:

                result = {
                    "student_id": None,
                    "name": "Unknown",
                    "college": None,
                    "score": float(best_score),
                    "match": False,
                    "face_index": face_index,
                    "face": face
                }

            results.append(result)

        return results

    def annotate_results(
        self,
        image,
        results
    ):

        annotated = image.copy()

        for result in results:

            face = result["face"]

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            if result["match"]:

                label = (
                    f'{result["name"]} '
                    f'| {result["student_id"]}'
                )

                score_label = (
                    f'Similarity: {result["score"]:.3f}'
                )

            else:

                label = "Unknown"

                if result["score"] >= 0:
                    score_label = (
                        f'Best similarity: '
                        f'{result["score"]:.3f}'
                    )
                else:
                    score_label = "No registered match"

            # Bounding box
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0)
                if result["match"]
                else (0, 0, 255),
                2
            )

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX

            label_size, _ = cv2.getTextSize(
                label,
                font,
                0.55,
                2
            )

            label_width = label_size[0]
            label_height = label_size[1]

            label_y = max(
                y - 10,
                label_height + 10
            )

            cv2.rectangle(
                annotated,
                (
                    x,
                    label_y - label_height - 8
                ),
                (
                    x + label_width + 10,
                    label_y + 4
                ),
                (0, 255, 0)
                if result["match"]
                else (0, 0, 255),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x + 5, label_y - 5),
                font,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Similarity score below bounding box
            cv2.putText(
                annotated,
                score_label,
                (x, y + h + 22),
                font,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated