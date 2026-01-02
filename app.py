# app.py
import streamlit as st
import subprocess

st.set_page_config(page_title="ClassEYE - Smart Student Attendance System")
st.title("ClassEYE - Smart Student Attendance System")

# ------------------- Session State -------------------
if "login_status" not in st.session_state:
    st.session_state.login_status = None  # None / "admin" / "class_end"

# ------------------- Navigation Dropdown -------------------
menu = ["Login"]
if st.session_state.login_status == "admin":
    menu = ["Enroll Students", "Logout"]
elif st.session_state.login_status == "class_end":
    menu = ["Start Attendance", "Logout"]

choice = st.selectbox("Menu", menu)

# ------------------- LOGIN SECTION -------------------
if st.session_state.login_status is None:
    st.subheader("Login")
    user_type = st.radio("Login as", ["Admin", "Class-End"])

    if user_type == "Admin":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login as Admin"):
            if username == "admin" and password == "admin123":
                st.session_state.login_status = "admin"
                st.success("Login Successful! You can now enroll students.")
            else:
                st.error("Invalid Username or Password")

    elif user_type == "Class-End":
        class_code = st.text_input("Enter Class Code")
        if st.button("Login as Class-End"):
            if class_code == "CLASS123":
                st.session_state.login_status = "class_end"
                st.success("Login Successful! You can start attendance.")
            else:
                st.error("Invalid Class Code")

# ------------------- ENROLL STUDENTS -------------------
if choice == "Enroll Students" and st.session_state.login_status == "admin":
    st.subheader("Enroll New Students")

    # Student input fields
    name = st.text_input("Name")
    enroll = st.text_input("Enrollment No.")
    batch = st.text_input("Batch Year")
    contact = st.text_input("Contact")
    branch = st.text_input("Branch")
    section = st.text_input("Section")
    group = st.text_input("Group")

    if st.button("Start Enrollment") and name and enroll:
        # Launch face_enroll.py with all info
        subprocess.Popen([
            "python", "face_enroll.py",
            name, enroll, batch, contact, branch, section, group
        ], shell=True)
        st.success(f"Enrollment started for {name}. Follow camera instructions to save face and details.")

# ------------------- START ATTENDANCE -------------------
elif choice == "Start Attendance" and st.session_state.login_status == "class_end":
    st.subheader("Start Attendance")

    class_code_input = st.text_input("Enter Class Code")
    if st.button("Start Attendance") and class_code_input:
        # Launch face_recognize.py with class code
        subprocess.Popen([
            "python", "face_recognize.py",
            class_code_input
        ], shell=True)
        st.info("Attendance started. Camera will run 10 minutes automatically.\nPress ESC in camera window to stop early.")

# ------------------- LOGOUT -------------------
if choice == "Logout":
    if st.button("Logout"):
        st.session_state.login_status = None
        st.experimental_rerun()
