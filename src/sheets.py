import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


def get_google_client():

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES
    )

    return gspread.authorize(credentials)


def get_spreadsheet():

    client = get_google_client()

    spreadsheet_name = st.secrets[
        "google_sheet_name"
    ]

    return client.open(spreadsheet_name)


def get_or_create_worksheet(
    spreadsheet,
    worksheet_name,
    headers
):

    try:

        worksheet = spreadsheet.worksheet(
            worksheet_name
        )

    except gspread.WorksheetNotFound:

        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=len(headers)
        )

        worksheet.append_row(headers)

    return worksheet


def initialize_sheets():

    spreadsheet = get_spreadsheet()

    students_headers = [
        "Student ID",
        "Name",
        "College",
        "Reference Photo"
    ]

    attendance_headers = [
        "Date",
        "Time",
        "Class",
        "Student ID",
        "Student Name",
        "Attendance",
        "Confidence"
    ]

    summary_headers = [
        "Student ID",
        "Student Name",
        "Total Classes",
        "Present",
        "Absent",
        "Attendance %"
    ]

    students_sheet = get_or_create_worksheet(
        spreadsheet,
        "Students",
        students_headers
    )

    attendance_sheet = get_or_create_worksheet(
        spreadsheet,
        "Attendance",
        attendance_headers
    )

    summary_sheet = get_or_create_worksheet(
        spreadsheet,
        "Summary",
        summary_headers
    )

    return (
        students_sheet,
        attendance_sheet,
        summary_sheet
    )


def add_student_to_sheet(
    student_id,
    name,
    college,
    photo
):

    students_sheet, _, _ = initialize_sheets()

    existing_rows = students_sheet.get_all_records()

    for row in existing_rows:

        if str(row.get("Student ID", "")).strip().lower() == \
                student_id.strip().lower():

            return False

    students_sheet.append_row(
        [
            student_id,
            name,
            college,
            photo
        ],
        value_input_option="USER_ENTERED"
    )

    return True


def attendance_session_exists(
    date,
    class_name
):

    _, attendance_sheet, _ = initialize_sheets()

    rows = attendance_sheet.get_all_records()

    target_date = str(date).strip()
    target_class = class_name.strip().lower()

    for row in rows:

        row_date = str(
            row.get("Date", "")
        ).strip()

        row_class = str(
            row.get("Class", "")
        ).strip().lower()

        if (
            row_date == target_date
            and row_class == target_class
        ):
            return True

    return False


def add_attendance_records(records):

    _, attendance_sheet, _ = initialize_sheets()

    rows = []

    for record in records:

        rows.append(
            [
                record["Date"],
                record["Time"],
                record["Class"],
                record["Student ID"],
                record["Student Name"],
                record["Attendance"],
                record["Confidence"]
            ]
        )

    if rows:

        attendance_sheet.append_rows(
            rows,
            value_input_option="USER_ENTERED"
        )


def update_summary_sheet():

    students_sheet, attendance_sheet, summary_sheet = \
        initialize_sheets()

    students = students_sheet.get_all_records()
    attendance = attendance_sheet.get_all_records()

    summary_rows = []

    # Unique class sessions
    sessions = set()

    for record in attendance:

        date = str(
            record.get("Date", "")
        ).strip()

        class_name = str(
            record.get("Class", "")
        ).strip().lower()

        if date and class_name:

            sessions.add(
                (date, class_name)
            )

    total_classes = len(sessions)

    for student in students:

        student_id = str(
            student.get("Student ID", "")
        ).strip()

        student_name = str(
            student.get("Name", "")
        ).strip()

        if not student_id:
            continue

        present_sessions = set()

        for record in attendance:

            record_student_id = str(
                record.get("Student ID", "")
            ).strip()

            attendance_status = str(
                record.get("Attendance", "")
            ).strip().lower()

            date = str(
                record.get("Date", "")
            ).strip()

            class_name = str(
                record.get("Class", "")
            ).strip().lower()

            if (
                record_student_id == student_id
                and attendance_status == "present"
                and date
                and class_name
            ):

                present_sessions.add(
                    (date, class_name)
                )

        present = len(present_sessions)

        absent = max(
            total_classes - present,
            0
        )

        if total_classes > 0:

            percentage = (
                present / total_classes
            ) * 100

        else:

            percentage = 0

        summary_rows.append(
            [
                student_id,
                student_name,
                total_classes,
                present,
                absent,
                round(percentage, 2)
            ]
        )

    # Clear old summary
    summary_sheet.clear()

    # Restore headers
    summary_sheet.append_row(
        [
            "Student ID",
            "Student Name",
            "Total Classes",
            "Present",
            "Absent",
            "Attendance %"
        ]
    )

    if summary_rows:

        summary_sheet.append_rows(
            summary_rows,
            value_input_option="USER_ENTERED"
        )