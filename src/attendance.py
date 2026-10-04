from pathlib import Path
import json
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

STUDENT_DIR = BASE_DIR / "data" / "students"
STUDENT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_FILE = BASE_DIR / "data" / "students.json"


def load_students():
    if not STUDENT_FILE.exists():
        return []

    with open(STUDENT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_students(students):
    with open(STUDENT_FILE, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4)


def register_student(
    student_id,
    name,
    college,
    college_email,
    branch,
    year,
    photo_bytes,
    face_feature
):
    students = load_students()

    # Prevent duplicate Student IDs
    for student in students:
        if student["student_id"].lower() == student_id.strip().lower():
            raise ValueError(
                f"Student ID '{student_id}' is already registered."
            )

    student_id = student_id.strip()

    # Save reference photo
    photo_path = STUDENT_DIR / f"{student_id}.jpg"

    image_array = np.frombuffer(
        photo_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to process the reference image."
        )

    cv2.imwrite(
        str(photo_path),
        image
    )

    # Save face embedding
    feature_path = STUDENT_DIR / f"{student_id}.npy"

    np.save(
        feature_path,
        face_feature
    )

    # Save student record
    students.append(
        {
            "student_id": student_id,
            "name": name.strip(),
            "college": college.strip(),
            "college_email": college_email.strip(),
            "branch": branch,
            "year": year,
            "photo": str(photo_path),
            "feature": str(feature_path)
        }
    )

    save_students(students)

    return student_id