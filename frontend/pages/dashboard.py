"""Dashboard page — document statistics and system health."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("📊 Dashboard")

try:
    # Fetch stats
    stats_resp = requests.get(f"{API_URL}/stats")
    health_resp = requests.get(f"{API_URL}/health")

    if stats_resp.status_code == 200:
        stats = stats_resp.json()

        # Key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Documents", stats["total_documents"])
        col2.metric("🧩 Total Chunks", stats["total_chunks"])
        col3.metric("💾 Collection", stats["collection_name"])

        # Documents by type
        st.subheader("Documents by Type")
        if stats["documents_by_type"]:
            type_icons = {"pdf": "📕 PDF", "docx": "📘 Word", "xlsx": "📗 Excel", "pptx": "📙 PowerPoint", "txt": "📄 Text"}
            for file_type, count in stats["documents_by_type"].items():
                label = type_icons.get(file_type, file_type)
                st.markdown(f"**{label}**: {count} document(s)")
        else:
            st.info("No documents uploaded yet.")

    # System health
    if health_resp.status_code == 200:
        health = health_resp.json()
        st.subheader("System Health")
        for service, status in health.get("services", {}).items():
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} **{service.title()}**: {'Running' if status else 'Not available'}")

except requests.ConnectionError:
    st.warning("⚠️ Cannot connect to the API. Make sure the backend is running.")
