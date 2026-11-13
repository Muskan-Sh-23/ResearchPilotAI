import streamlit as st
from pypdf import PdfReader

from utils.text_splitter import split_text
from utils.embeddings import create_embeddings
from utils.vector_store import create_vector_store, search_vector_store
from utils.llm import generate_answer_stream

TOP_K = 5

st.set_page_config(page_title="ResearchPilot AI", page_icon="🔬")
st.title("🔬 ResearchPilot AI")
st.caption("Upload a research paper and ask questions about it.")


# ---------- Indexing (runs once per uploaded file) ----------
def index_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    chunk_records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            for chunk in split_text(text):
                chunk_records.append({"page": page_number, "text": chunk})

    if not chunk_records:
        return None

    chunks = [c["text"] for c in chunk_records]
    embeddings = create_embeddings(chunks)
    vector_store = create_vector_store(embeddings)

    return {
        "name": uploaded_file.name,
        "num_pages": len(reader.pages),
        "chunk_records": chunk_records,
        "vector_store": vector_store,
    }


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your paper")
    uploaded_file = st.file_uploader("Upload your research paper", type="pdf")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_file:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# Re-index only when a NEW file is uploaded
if st.session_state.get("doc", {}).get("name") != uploaded_file.name:
    with st.spinner("Reading and indexing your paper..."):
        try:
            doc = index_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't process this PDF: {e}")
            st.stop()

    if doc is None:
        st.error(
            "No readable text found. This is probably a scanned PDF "
            "(images of pages). Please try a text-based PDF."
        )
        st.stop()

    st.session_state.doc = doc
    st.session_state.messages = []  # fresh chat for a new paper

doc = st.session_state.doc
chunk_records = doc["chunk_records"]

st.success("Document indexed successfully!")
st.write(f"📄 **{doc['name']}**")
st.caption(f"{doc['num_pages']} pages • {len(chunk_records)} searchable sections")

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Pages searched: " + ", ".join(f"p. {p}" for p in msg["sources"]))

# ---------- New question ----------
query = st.chat_input("Ask a question about the paper...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the paper..."):
                query_embedding = create_embeddings([query])
                distances, indices = search_vector_store(
                    doc["vector_store"], query_embedding, k=TOP_K
                )

                # FAISS returns -1 when fewer than k results exist
                hits = [
                    (int(idx), float(dist))
                    for idx, dist in zip(indices[0], distances[0])
                    if idx >= 0
                ]
                # Order by page so the context reads in document order
                hits.sort(key=lambda h: chunk_records[h[0]]["page"])

                context = "\n\n".join(
                    f"[Page {chunk_records[i]['page']}]\n{chunk_records[i]['text']}"
                    for i, _ in hits
                )

            source_pages = sorted({chunk_records[i]["page"] for i, _ in hits})

            # Stream the answer as it is generated
            status = st.empty()
            status.info("⏳ Generating answer...")

            def _stream():
                first = True
                for piece in generate_answer_stream(query, context):
                    if first:
                        status.empty()
                        first = False
                    yield piece

            answer = st.write_stream(_stream())
            status.empty()

            if not answer or not answer.strip():
                st.warning("The model returned an empty answer. Please try again.")
            st.caption("Pages searched: " + ", ".join(f"p. {p}" for p in source_pages))

            with st.expander("🔎 Retrieved evidence"):
                for rank, (i, dist) in enumerate(hits, start=1):
                    st.markdown(f"**Match {rank} — page {chunk_records[i]['page']}**")
                    st.write(chunk_records[i]["text"])
                    st.caption(f"Distance: {dist:.4f} (lower = closer)")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": source_pages}
            )

        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                st.error("The AI service is busy or its quota is used up. Please try again in a minute.")
            elif "503" in msg or "500" in msg:
                st.error("The AI service is temporarily overloaded. Please try again shortly.")
            else:
                st.error(f"Something went wrong: {msg}")