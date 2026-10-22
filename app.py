import cv2
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

# ---------------- PATHS ----------------
DATASET_PATH = "dataset/faces"
STUDENT_FILE = "dataset/students.csv"
ATTENDANCE_FILE = "dataset/attendance.csv"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# create files if not exist
if not os.path.exists(STUDENT_FILE):
    pd.DataFrame(columns=["id", "name"]).to_csv(STUDENT_FILE, index=False)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["id", "name", "date", "time"]).to_csv(ATTENDANCE_FILE, index=False)

# ---------------- FACE CASCADE ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- REGISTER STUDENT ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_id = int(request.form["id"])
        name = request.form["name"]

        cam = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                filename = f"{student_id}_{name}_{count}.jpg"
                cv2.imwrite(os.path.join(DATASET_PATH, filename), face)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Register Face - Press Q", frame)

            if cv2.waitKey(1) & 0xFF == ord("q") or count >= 20:
                break

        cam.release()
        cv2.destroyAllWindows()

        df = pd.read_csv(STUDENT_FILE)
        df.loc[len(df)] = [student_id, name]
        df.to_csv(STUDENT_FILE, index=False)

        return redirect(url_for("index"))

    return render_template("register.html")

# ---------------- MARK ATTENDANCE ----------------
@app.route("/mark")
def mark():
    students = pd.read_csv(STUDENT_FILE)

    faces_data = []
    labels = []

    for file in os.listdir(DATASET_PATH):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(DATASET_PATH, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            faces_data.append(img)
            labels.append(int(file.split("_")[0]))

    labels = np.array(labels)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces_data, labels)

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label, confidence = model.predict(face_img)
            student = students[students["id"] == label]

            if not student.empty and confidence < 70:
                name = student.iloc[0]["name"]
                mark_attendance(label, name)
                text = name
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Attendance - Press Q", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    return redirect(url_for("attendance"))

# ---------------- ATTENDANCE LOGIC ----------------
def mark_attendance(student_id, name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)
    mask = (df["id"] == student_id) & (df["date"] == today)

    if mask.any():
        df.loc[mask, "time"] = time_now
    else:
        df.loc[len(df)] = [student_id, name, today, time_now]

    df.to_csv(ATTENDANCE_FILE, index=False)

# ---------------- VIEW ATTENDANCE ----------------
@app.route("/attendance")
def attendance():
    df = pd.read_csv(ATTENDANCE_FILE)
    records = df.to_dict(orient="records")
    return render_template("attendance.html", records=records)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)