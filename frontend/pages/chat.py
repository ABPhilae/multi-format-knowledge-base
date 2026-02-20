"""Chat page — conversational Q&A with source citations."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("💬 Chat with Your Documents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# New conversation button
if st.sidebar.button("🔄 New Conversation"):
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    icon = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "pptx": "📙"}.get(source.get("file_type", ""), "📄")
                    st.markdown(f"{icon} **{source['source']}**")
                    st.caption(source["content"][:200])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])
                    st.session_state.conversation_id = data["conversation_id"]

                    # Show sources
                    if data.get("sources"):
                        with st.expander("📎 Sources"):
                            for source in data["sources"]:
                                icon = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "pptx": "📙"}.get(source.get("file_type", ""), "📄")
                                st.markdown(f"{icon} **{source['source']}**")
                                st.caption(source["content"][:200])

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    })
                else:
                    st.error(f"Error: {response.text}")

            except requests.ConnectionError:
                st.error("Cannot connect to the API. Make sure the backend is running.")
