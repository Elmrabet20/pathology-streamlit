import cv2
from collections import Counter
from ultralytics import RTDETR
import streamlit as st

model = RTDETR("weights/best.pt")  # Modifie avec ton vrai chemin

def process_video_with_live(input_path, output_path, selected_classes, progress_callback=None, start_frame=0):
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if "results" not in st.session_state:
        st.session_state["results"] = []
    elif start_frame == 0:
        st.session_state["results"].clear()

    class_counter = Counter()
    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)[0]
        label, score = "N/A", 0.0

        for box in results.boxes.data:
            class_id = int(box[5])
            label = model.names[class_id]
            score = float(box[4])
            if label in selected_classes:
                class_counter[label] += 1

        # Résumé frame par frame
        st.session_state["results"].append({
            "frame": current_frame,
            "class": label,
            "score": round(score, 2)
        })

        out.write(results.plot())

        if progress_callback:
            progress_callback(
                current_frame / total_frames,
                frame_image=results.plot(),
                frame_idx=current_frame,
                total=total_frames,
                label=label,
                score=score
            )

        current_frame += 1

        if st.session_state.get("app_state", {}).get("arret"):
            break

    cap.release()
    out.release()

    st.session_state["__detection_done__"] = True
    st.session_state["__class_counter__"] = class_counter
    return class_counter
