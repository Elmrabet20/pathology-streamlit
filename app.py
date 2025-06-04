# app.py
import streamlit as st
import os
import uuid
import tempfile
import pandas as pd
from detection import process_video_with_live

st.set_page_config(page_title="🩺 Pathology Detection App", layout="wide")

st.sidebar.title("🩺 Menu")
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🔬 Analyse Vidéo", "ℹ️ À propos"])

if "app_state" not in st.session_state:
    st.session_state["app_state"] = {
        "en_cours": False,
        "arret": False,
        "class_counts": None,
        "results": [],
        "__detection_done__": False,
        "input_path": None,
        "output_path": None,
        "last_frame": 0,
        "status": "🟡 En attente"
    }
state = st.session_state["app_state"]

st.sidebar.markdown("---")
st.sidebar.write(f"**Statut détection :** {state['status']}")

if page == "🏠 Accueil":
    st.title("🩺 Pathology Detection App")
    st.caption("Projet de Fin d'Études - Master AISD")
    st.markdown("Bienvenue dans la plateforme de **détection automatique de pathologies** par vidéos.")

elif page == "🔬 Analyse Vidéo":
    st.title("🔬 Analyse en temps réel")
    uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video and not state["en_cours"] and not state["__detection_done__"]:
        if st.button("▶️ Start Detection"):
            state["en_cours"] = True
            state["arret"] = False
            state["results"] = []
            input_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
            with open(input_path, "wb") as f:
                f.write(uploaded_video.read())
            state["input_path"] = input_path
            state["output_path"] = input_path.replace(".mp4", "_annotated.mp4")
            state["status"] = "🔄 En cours"
            st.rerun()

    if state["en_cours"] and not state["__detection_done__"]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ Arrêter la détection"):
                state["arret"] = True
                state["status"] = "⏸️ Arrêtée"
        with col2:
            if state["arret"] and st.button("🔁 Reprendre la détection"):
                state["arret"] = False
                state["en_cours"] = True
                state["status"] = "🔄 Reprise"
                st.rerun()

        stframe = st.empty()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        metrics = st.empty()
        resume_placeholder = st.empty()

        def update_progress(p, frame_image=None, frame_idx=0, total=100, label="N/A", score=0.0):
            if state["arret"]:
                state["last_frame"] = frame_idx
                state["en_cours"] = False
                raise st.StopException

            progress_bar.progress(min(p, 1.0))
            status_text.text(f"Frame {frame_idx}/{total} - {label} ({score:.2f})")
            if frame_image is not None:
                stframe.image(frame_image, channels="BGR", caption=f"Frame {frame_idx}")
                metrics.metric("Dernière classe détectée", label, delta=f"{score:.2f}")

            if st.session_state["results"]:
                df = pd.DataFrame(st.session_state["results"])
                df.columns = ["Frame", "Classe détectée", "Score de confiance"]
                df.index.name = "#"
                resume_placeholder.dataframe(df.tail(10), use_container_width=True)

        class_counts = process_video_with_live(
            state["input_path"],
            state["output_path"],
            selected_classes=["TP", "TNP"],
            progress_callback=update_progress,
            start_frame=state["last_frame"]
        )

        st.success("✅ Analyse terminée")
        state["class_counts"] = class_counts
        state["en_cours"] = False
        state["__detection_done__"] = True
        state["status"] = "✅ Terminée"

        with open(state["output_path"], "rb") as f:
            st.download_button("⬇️ Télécharger la vidéo annotée", f, file_name="annotated_output.mp4")


elif page == "ℹ️ À propos":
    st.title("À propos du projet")
    st.markdown("Projet de FIn d'Études - FSTT")
    st.markdown("Ce projet applique la détection automatique sur vidéos médicales via RT-DETR.")

st.markdown("---")
st.markdown("© 2025 - Projet de Fin d'Études réalisé par **ELM'RABET Hanae** | Université FSTT - Master AISD")
