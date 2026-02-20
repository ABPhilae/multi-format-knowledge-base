"""Evaluation page — run RAGAS evaluation and display scores."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🎯 RAG Quality Evaluation")
st.markdown("""
Run RAGAS evaluation to measure the quality of your RAG system across 4 dimensions.
This process asks 20 test questions and scores the answers.

**Score guide:** 0.9+ = Excellent | 0.7-0.9 = Good | 0.5-0.7 = Needs work | Below 0.5 = Problems
""")

if st.button("▶️ Run Evaluation", type="primary"):
    with st.spinner("Running RAGAS evaluation... This may take 3-5 minutes."):
        try:
            response = requests.post(f"{API_URL}/evaluate", timeout=600)

            if response.status_code == 200:
                results = response.json()

                if "error" in results:
                    st.error(f"Evaluation error: {results['error']}")
                else:
                    # Overall score
                    overall = results["overall_score"]
                    if overall >= 0.9:
                        st.success(f"🏆 Overall Score: {overall:.2%} — Excellent!")
                    elif overall >= 0.7:
                        st.info(f"✅ Overall Score: {overall:.2%} — Good")
                    elif overall >= 0.5:
                        st.warning(f"⚠️ Overall Score: {overall:.2%} — Needs improvement")
                    else:
                        st.error(f"❌ Overall Score: {overall:.2%} — Significant issues")

                    # Individual metrics
                    st.subheader("Detailed Metrics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Faithfulness", f"{results['faithfulness']:.2%}",
                                  help="Does the answer stick to what the documents say?")
                        st.metric("Context Precision", f"{results['context_precision']:.2%}",
                                  help="Are the retrieved chunks actually relevant?")

                    with col2:
                        st.metric("Answer Relevancy", f"{results['answer_relevancy']:.2%}",
                                  help="Does the answer address the question?")
                        st.metric("Context Recall", f"{results['context_recall']:.2%}",
                                  help="Are all needed chunks being retrieved?")

                    st.caption(f"Evaluated {results['questions_evaluated']} questions at {results['timestamp']}")
            else:
                st.error(f"Evaluation failed: {response.text}")

        except requests.ConnectionError:
            st.error("Cannot connect to the API.")
        except requests.Timeout:
            st.error("Evaluation timed out. Try again with fewer test questions.")

st.markdown("---")
st.markdown("""
### How to Read the Scores

| Low Score | Likely Problem | Fix |
|-----------|---------------|-----|
| Low Faithfulness | LLM is hallucinating | Improve prompt, lower temperature |
| Low Answer Relevancy | Not answering the question asked | Review prompt template |
| Low Context Precision | Too many irrelevant chunks retrieved | Add re-ranking, increase similarity threshold |
| Low Context Recall | Missing relevant chunks | Improve chunking strategy, increase top_k |
""")
