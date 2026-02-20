"""Upload page — drag-and-drop document upload with real-time status."""
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.title("📤 Upload Documents")
st.markdown("Upload any supported document type. Processing happens in the background.")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=["pdf", "docx", "xlsx", "pptx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                st.info(f"📄 {uploaded_file.name} — Processing started (Job: {job_id[:8]}...)")

                # Poll for completion
                progress_bar = st.progress(0)
                status = "pending"
                attempts = 0
                while status not in ("completed", "failed") and attempts < 60:
                    time.sleep(2)
                    attempts += 1
                    progress_bar.progress(min(attempts * 5, 95))
                    status_resp = requests.get(f"{API_URL}/documents/{job_id}/status")
                    if status_resp.status_code == 200:
                        status = status_resp.json()["status"]

                progress_bar.progress(100)

                if status == "completed":
                    st.success(f"✅ {uploaded_file.name} — Processing complete!")
                else:
                    st.error(f"❌ {uploaded_file.name} — Processing failed")
            else:
                st.error(f"❌ Upload failed: {response.text}")

# Show current documents
st.markdown("---")
st.subheader("📁 Current Documents")
try:
    docs_resp = requests.get(f"{API_URL}/documents")
    if docs_resp.status_code == 200:
        docs = docs_resp.json()
        if docs:
            for doc in docs:
                icon = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "pptx": "📙", "txt": "📄"}.get(doc["file_type"], "📄")
                st.markdown(f"{icon} **{doc['filename']}** — {doc['chunk_count']} chunks — {doc['status']}")
        else:
            st.info("No documents uploaded yet.")
except requests.ConnectionError:
    st.warning("⚠️ Cannot connect to the API. Make sure the backend is running.")
