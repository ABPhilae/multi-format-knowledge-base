"""
Multi-Format Knowledge Base — Streamlit Frontend.

Multi-page app with:
- Chat: Ask questions with conversation memory
- Upload: Drag-and-drop any file type
- Dashboard: Document stats and collection info
- Evaluation: RAGAS quality scores
"""
import streamlit as st

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📚 Knowledge Base")
st.sidebar.markdown("Multi-Format Document Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("LangChain • LlamaIndex • RAGAS")
st.sidebar.markdown("---")
st.sidebar.markdown("Navigate using the pages above ☝️")

st.title("📚 Multi-Format Knowledge Base")
st.markdown("""
Welcome to the Multi-Format Knowledge Base. Upload any document type
and ask questions in natural language.

**Supported file types:** PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), Text

**Pages:**
- **💬 Chat** — Ask questions about your documents
- **📤 Upload** — Upload new documents
- **📊 Dashboard** — View document statistics
- **🎯 Evaluation** — See RAGAS quality scores

Use the sidebar to navigate between pages.
""")
