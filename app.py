import streamlit as st
import cv2
import numpy as np
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO

from src.face_engine import FaceEngine
from src.attendance import register_student, load_students


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ATTENDANCE_DIR = BASE_DIR / "data" / "attendance"
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Face Embedding Matching",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }

    .hero p {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 0;
    }

    .card {
        padding: 1.4rem;
        border-radius: 16px;
        background-color: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .metric-title {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 700;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOCAL ATTENDANCE FUNCTIONS
# ============================================================

def save_attendance_records(records):

    df = pd.DataFrame(records)

    if ATTENDANCE_FILE.exists():

        existing_df = pd.read_csv(
            ATTENDANCE_FILE
        )

        df = pd.concat(
            [existing_df, df],
            ignore_index=True
        )

    df.to_csv(
        ATTENDANCE_FILE,
        index=False
    )


def attendance_session_exists(date_string, class_name):

    if not ATTENDANCE_FILE.exists():
        return False

    df = pd.read_csv(ATTENDANCE_FILE)

    if df.empty:
        return False

    target_date = str(date_string).strip()
    target_class = str(class_name).strip().lower()

    return bool(
        ((df["Date"].astype(str).str.strip() == target_date) &
         (df["Class"].astype(str).str.strip().str.lower() == target_class)).any()
    )


def load_attendance_records():

    if not ATTENDANCE_FILE.exists():

        return pd.DataFrame(
            columns=[
                "Date",
                "Time",
                "Class",
                "Student ID",
                "Student Name",
                "Attendance",
                "Confidence"
            ]
        )

    return pd.read_csv(
        ATTENDANCE_FILE
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🎓 Face Embedding Matching</h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Computer Vision Attendance System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Student Registration",
            "Take Attendance",
            "Attendance History"
        ]
    )

    st.divider()

    st.caption(
        "Deep Learning • Computer Vision • Face Recognition"
    )

    st.info(
        "Local mode: attendance is stored in data/attendance/attendance.csv."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Face Embedding Matching Engine</h1>
        <p>
            Deep Facial Representation, Embedding Extraction
            & Similarity-Based Identity Recognition
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">System Dashboard</div>',
        unsafe_allow_html=True
    )

    students = load_students()
    attendance_df = load_attendance_records()

    total_students = len(students)

    total_sessions = (
        attendance_df["Date"].nunique()
        if not attendance_df.empty
        else 0
    )

    total_present = (
        (
            attendance_df["Attendance"] == "Present"
        ).sum()
        if not attendance_df.empty
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Registered Students
                </div>
                <div class="metric-value">
                    {total_students}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Attendance Sessions
                </div>
                <div class="metric-value">
                    {total_sessions}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">
                    Total Present Records
                </div>
                <div class="metric-value">
                    {total_present}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.info(
        """
        **System Pipeline**

        Student Registration → Face Detection → Face Embedding →
        Classroom Image → Multi-Face Detection →
        Similarity Matching → Present / Absent →
        Local Attendance Database (CSV)
        """
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

elif page == "Student Registration":

    st.markdown(
        '<div class="section-title">Student Registration</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Register a student using a unique Student ID "
        "and a clear reference photograph."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # STUDENT DETAILS
    # --------------------------------------------------------

    with col1:

        st.markdown("### Student Details")

        student_id = st.text_input(
            "Student ID",
            placeholder="Example: K12345"
        ).strip().upper()

        name = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )

        college = st.text_input(
            "College",
            placeholder="Enter college name"
        )

        st.caption(
            "Student ID format: K + exactly 5 digits."
        )

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    with col2:

        st.markdown("### Reference Photograph")

        photo_source = st.radio(
            "Choose photo source",
            [
                "📷 Camera",
                "📁 Upload from Device"
            ],
            horizontal=True
        )

        photo = None

        if photo_source == "📷 Camera":

            photo = st.camera_input(
                "Capture Reference Photo"
            )

        else:

            photo = st.file_uploader(
                "Upload Reference Photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                help=(
                    "Upload a clear frontal photograph "
                    "containing one person."
                )
            )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    if photo:

        st.markdown("### Photo Preview")

        st.image(
            photo,
            caption="Selected Reference Photograph",
            width=300
        )

    st.divider()

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    if st.button(
        "Register Student",
        type="primary",
        use_container_width=True
    ):

        if (
            not student_id
            or not name
            or not college
            or photo is None
        ):

            st.error(
                "Please complete Student ID, Name, College "
                "and Reference Photo."
            )

        elif not re.fullmatch(
            r"K\d{5}",
            student_id
        ):

            st.error(
                "Invalid Student ID. Please use the format "
                "Kxxxxx. Example: K12345"
            )

        else:

            try:

                # --------------------------------------------
                # CHECK DUPLICATE
                # --------------------------------------------

                students = load_students()

                duplicate = any(
                    student["student_id"].upper()
                    == student_id.upper()
                    for student in students
                )

                if duplicate:

                    st.error(
                        f"Student ID '{student_id}' "
                        "is already registered."
                    )

                else:

                    # ----------------------------------------
                    # INITIALIZE ENGINE
                    # ----------------------------------------

                    with st.spinner(
                        "Loading face recognition engine..."
                    ):

                        engine = FaceEngine()

                    # ----------------------------------------
                    # READ IMAGE
                    # ----------------------------------------

                    image_array = np.frombuffer(
                        photo.getvalue(),
                        dtype=np.uint8
                    )

                    image = cv2.imdecode(
                        image_array,
                        cv2.IMREAD_COLOR
                    )

                    if image is None:

                        raise ValueError(
                            "Unable to read the selected image."
                        )

                    # ----------------------------------------
                    # DETECT FACE
                    # ----------------------------------------

                    with st.spinner(
                        "Detecting face..."
                    ):

                        faces = engine.detect_faces(
                            image
                        )

                    if len(faces) == 0:

                        st.error(
                            "No face detected. "
                            "Please use a clear frontal photograph."
                        )

                    elif len(faces) > 1:

                        st.error(
                            "Multiple faces detected. "
                            "Please use a photograph containing "
                            "only one person."
                        )

                    else:

                        # ------------------------------------
                        # GENERATE EMBEDDING
                        # ------------------------------------

                        with st.spinner(
                            "Generating face embedding..."
                        ):

                            feature = engine.get_face_feature(
                                image,
                                faces[0]
                            )

                        # ------------------------------------
                        # SAVE LOCAL STUDENT DATA
                        # ------------------------------------

                        registered_id = register_student(
                            student_id=student_id,
                            name=name,
                            college=college,
                            photo_bytes=photo.getvalue(),
                            face_feature=feature
                        )

                        # ------------------------------------
                        # SUCCESS
                        # ------------------------------------

                        st.success(
                            f"Student {registered_id} "
                            "registered successfully."
                        )

                        st.balloons()

            except Exception as error:

                st.error(
                    f"Registration failed: {error}"
                )


# ============================================================
# TAKE ATTENDANCE
# ============================================================

elif page == "Take Attendance":

    st.markdown(
        '<div class="section-title">Take Classroom Attendance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Detect and identify registered students from "
        "a classroom photograph."
    )

    st.divider()

    # --------------------------------------------------------
    # CLASS DETAILS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        subject = st.text_input(
            "Class / Subject",
            placeholder="Example: Deep Learning"
        )

    with col2:

        attendance_date = st.date_input(
            "Attendance Date"
        )

    # --------------------------------------------------------
    # IMAGE SOURCE
    # --------------------------------------------------------

    classroom_source = st.radio(
        "Classroom Image Source",
        [
            "📷 Camera",
            "📁 Upload Image"
        ],
        horizontal=True
    )

    classroom_image = None

    if classroom_source == "📷 Camera":

        classroom_image = st.camera_input(
            "Capture Classroom Image"
        )

    else:

        classroom_image = st.file_uploader(
            "Upload Classroom Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help=(
                "Use a clear classroom photograph "
                "where student faces are visible."
            )
        )

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    if classroom_image:

        st.image(
            classroom_image,
            caption="Classroom Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        if st.button(
            "🔍 Analyze Attendance",
            type="primary",
            use_container_width=True
        ):

            students = load_students()

            if not students:

                st.warning(
                    "No students are registered yet. "
                    "Please register students first."
                )

            elif not subject.strip():

                st.warning(
                    "Please enter the class / subject name."
                )

            else:

                date_string = attendance_date.strftime("%Y-%m-%d")

                if attendance_session_exists(date_string, subject.strip()):

                    st.error(
                        f"Attendance already exists for {date_string} — {subject.strip()}."
                    )

                else:

                    try:

                        # ----------------------------------------
                            # READ CLASSROOM IMAGE
                        # ----------------------------------------

                        image_array = np.frombuffer(
                            classroom_image.getvalue(),
                            dtype=np.uint8
                        )

                        classroom_cv = cv2.imdecode(
                            image_array,
                            cv2.IMREAD_COLOR
                        )

                        if classroom_cv is None:

                            raise ValueError(
                                "Unable to read classroom image."
                            )

                        # ----------------------------------------
                        # LOAD ENGINE
                        # ----------------------------------------

                        with st.spinner(
                            "Loading deep-learning recognition engine..."
                        ):

                            engine = FaceEngine()

                        # ----------------------------------------
                        # MULTI-FACE RECOGNITION
                        # ----------------------------------------

                        with st.spinner(
                            "Detecting faces and matching identities..."
                        ):

                            recognition_results = (
                                engine.recognize_students(
                                    classroom_cv,
                                    students,
                                    threshold=0.363
                                )
                            )

                        # ----------------------------------------
                        # COLLECT MATCHES
                        # ----------------------------------------

                        matched_students = {}

                        for result in recognition_results:

                            if result["match"]:

                                student_id = result["student_id"]

                                # Keep highest score if
                                # same student appears multiple times
                                if (
                                    student_id not in matched_students
                                    or result["score"]
                                    > matched_students[
                                        student_id
                                    ]["score"]
                                ):

                                    matched_students[
                                        student_id
                                    ] = result

                        # ----------------------------------------
                        # CREATE ATTENDANCE RECORDS
                        # ----------------------------------------

                        current_time = datetime.now().strftime(
                            "%H:%M:%S"
                        )

                        date_string = (
                            attendance_date.strftime(
                                "%Y-%m-%d"
                            )
                        )

                        records = []

                        for student in students:

                            student_id = student["student_id"]

                            if student_id in matched_students:

                                result = matched_students[
                                    student_id
                                ]

                                attendance = "Present"

                                confidence = round(
                                    result["score"],
                                    4
                                )

                            else:

                                attendance = "Absent"
                                confidence = ""

                            records.append(
                                {
                                    "Date": date_string,
                                    "Time": current_time,
                                    "Class": subject.strip(),
                                    "Student ID": student_id,
                                    "Student Name": student["name"],
                                    "Attendance": attendance,
                                    "Confidence": confidence
                                }
                            )

                        # ----------------------------------------
                        # SAVE LOCAL ATTENDANCE
                        # ----------------------------------------

                        save_attendance_records(
                            records
                        )

                        # ----------------------------------------
                        # SUCCESS
                        # ----------------------------------------

                        st.success(
                            "Attendance analysis completed and saved locally."
                        )

                        # ----------------------------------------
                        # METRICS
                        # ----------------------------------------

                        total_detected = len(
                            recognition_results
                        )

                        total_present = len(
                            matched_students
                        )

                        total_absent = (
                            len(students)
                            - total_present
                        )

                        metric1, metric2, metric3 = st.columns(3)

                        with metric1:

                            st.metric(
                                "Faces Detected",
                                total_detected
                            )

                        with metric2:

                            st.metric(
                                "Students Present",
                                total_present
                            )

                        with metric3:

                            st.metric(
                                "Students Absent",
                                total_absent
                            )

                        # ----------------------------------------
                        # FACE MATCHING RESULTS
                        # ----------------------------------------

                        st.markdown(
                            "### Face Matching Results"
                        )

                        if recognition_results:

                            result_rows = []

                            for result in recognition_results:

                                result_rows.append(
                                    {
                                        "Detected Face":
                                            result["face_index"] + 1,

                                        "Identity":
                                            result["name"],

                                        "Student ID":
                                            result["student_id"]
                                            or "—",

                                        "Similarity":
                                            round(
                                                result["score"],
                                                4
                                            ),

                                        "Status":
                                            (
                                                "Matched"
                                                if result["match"]
                                                else "Unknown"
                                            )
                                    }
                                )

                            st.dataframe(
                                result_rows,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.warning(
                                "No faces detected."
                            )

                        # ----------------------------------------
                        # ATTENDANCE TABLE
                        # ----------------------------------------

                        st.markdown(
                            "### Attendance Record"
                        )

                        attendance_result_df = pd.DataFrame(
                            records
                        )

                        st.dataframe(
                            attendance_result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                    except Exception as error:

                        st.error(
                            f"Attendance analysis failed: {error}"
                        )


# ============================================================
# ATTENDANCE HISTORY
# ============================================================

st.header("📊 Attendance History")

attendance_df = load_attendance_records()

if attendance_df.empty:

    st.info("No attendance records available yet.")

else:

    attendance_df["Date"] = pd.to_datetime(
        attendance_df["Date"],
        errors="coerce"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_class = st.selectbox(
            "Class",
            ["All"] + sorted(
                attendance_df["Class"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with col2:

        min_date = attendance_df["Date"].min().date()
        max_date = attendance_df["Date"].max().date()

        selected_dates = st.date_input(
            "Date Range",
            value=(min_date, max_date)
        )

    with col3:

        selected_student = st.selectbox(
            "Student",
            ["All"] + sorted(
                attendance_df["Student Name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    filtered_df = attendance_df.copy()

    if selected_class != "All":

        filtered_df = filtered_df[
            filtered_df["Class"] == selected_class
        ]

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = filtered_df[
            (
                filtered_df["Date"].dt.date >= start_date
            )
            &
            (
                filtered_df["Date"].dt.date <= end_date
            )
        ]

    if selected_student != "All":

        filtered_df = filtered_df[
            filtered_df["Student Name"] == selected_student
        ]

    total_records = len(filtered_df)

    present_count = len(
        filtered_df[
            filtered_df["Attendance"]
            .astype(str)
            .str.lower()
            == "present"
        ]
    )

    absent_count = len(
        filtered_df[
            filtered_df["Attendance"]
            .astype(str)
            .str.lower()
            == "absent"
        ]
    )

    attendance_percentage = (
        present_count / total_records * 100
        if total_records > 0
        else 0
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Records",
        total_records
    )

    m2.metric(
        "Present",
        present_count
    )

    m3.metric(
        "Absent",
        absent_count
    )

    m4.metric(
        "Attendance %",
        f"{attendance_percentage:.1f}%"
    )

    st.divider()

    display_df = filtered_df.copy()

    display_df["Date"] = display_df[
        "Date"
    ].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Student Attendance Summary")

    summary = (
        filtered_df
        .groupby(
            ["Student ID", "Student Name"]
        )
        .agg(
            Total=("Attendance", "count"),
            Present=(
                "Attendance",
                lambda x: (
                    x.astype(str)
                    .str.lower()
                    .eq("present")
                    .sum()
                )
            )
        )
        .reset_index()
    )

    summary["Absent"] = (
        summary["Total"]
        - summary["Present"]
    )

    summary["Attendance %"] = (
        summary["Present"]
        / summary["Total"]
        * 100
    ).round(2)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # Excel export
    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        display_df.to_excel(
            writer,
            sheet_name="Attendance",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

    excel_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Attendance Excel",
        data=excel_buffer,
        file_name="attendance_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

