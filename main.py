import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import threading
import tempfile
import os
from ultralytics import RTDETR
from collections import Counter

# Charger le modèle
model = RTDETR("weights/best.pt")

# État global
output_video_path = os.path.join(tempfile.gettempdir(), "output_live.mp4")
detected_classes = Counter()
record_video = st.checkbox("💾 Enregistrer la vidéo")
selected_classes = st.multiselect("🎯 Classes à détecter :", options=model.names.values(), default=list(model.names.values()))

# Initialisation vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None
lock = threading.Lock()

st.title("🎥 RT-DETR - Live Object Detection")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.writer = None
        self.started = False

    def transform(self, frame):
        global video_writer, detected_classes
        img = frame.to_ndarray(format="bgr24")

        # Détection
        results = model.predict(img)[0]
        annotated = results.plot()

        # Filtrage
        filtered_boxes = []
        for box in results.boxes.data:
            class_id = int(box[5])
            class_name = model.names[class_id]
            if class_name in selected_classes:
                detected_classes[class_name] += 1

        # Enregistrement si activé
        if record_video:
            with lock:
                if video_writer is None:
                    height, width = annotated.shape[:2]
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
                video_writer.write(annotated)

        return annotated

    def __del__(self):
        if video_writer:
            with lock:
                video_writer.release()

# Lancer le flux webcam
ctx = webrtc_streamer(
    key="realtime-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Afficher les classes détectées
st.subheader("📊 Classes détectées (compteur en temps réel)")
st.json(dict(detected_classes))

# Télécharger la vidéo
if record_video and os.path.exists(output_video_path):
    with open(output_video_path, "rb") as file:
        st.download_button("⬇️ Télécharger la vidéo annotée", file, file_name="rt_detr_annotated.mp4")
