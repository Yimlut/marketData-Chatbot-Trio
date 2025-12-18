import os
import time
from pathlib import Path
import streamlit as st
from openai import OpenAI

# =========================
# Basic config
# =========================
st.set_page_config(page_title="AT&T 10-K – Q&A (Responses + File Search)",
                   page_icon="📄", layout="centered")
st.title("📄 AT&T 10-K – Q&A (Managed Retrieval via Responses API)")
st.caption("OpenAI Responses API + File Search (Vector Stores). API key comes from environment.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. On Windows run:  setx OPENAI_API_KEY your_key_here  (then open a NEW terminal).")
    st.stop()

client = OpenAI()

PDF_DEFAULT = "ATT 10K.pdf"                     # your 109-page / ~3.5MB file
VECTOR_ID_FILE = Path("vector_store_id.txt")    # persists vector store id


# =========================
# Helpers
# =========================
def read_vector_id() -> str | None:
    if VECTOR_ID_FILE.exists():
        t = VECTOR_ID_FILE.read_text(encoding="utf-8").strip()
        return t or None
    return None

def write_vector_id(vs_id: str) -> None:
    VECTOR_ID_FILE.write_text(vs_id.strip(), encoding="utf-8")


# Session state
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content)]
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = read_vector_id()


# =========================
# Sidebar — Vector Store management
# =========================
with st.sidebar:
    st.subheader("Knowledge base")
    st.write("Create a **Vector Store** and attach your PDF. The ID is saved locally in `vector_store_id.txt`, "
             "so you don’t re-embed on every run.")

    pdf_path_text = st.text_input("PDF path", value=PDF_DEFAULT)

    c1, c2 = st.columns(2)
    create_btn = c1.button("📥 Create / Reindex")
    load_btn   = c2.button("🔗 Load Saved ID")

    manual_vs = st.text_input("Paste existing Vector Store ID (optional)")
    c3, c4 = st.columns(2)
    save_btn  = c3.button("💾 Save ID")
    clear_btn = c4.button("🧹 Clear saved ID")

    check_btn = st.button("🔎 Check indexing status")

    if load_btn:
        vs = read_vector_id()
        st.session_state.vector_store_id = vs
        if vs:
            st.success(f"Loaded Vector Store: {vs}")
        else:
            st.warning("No saved vector_store_id.txt found.")

    if save_btn and manual_vs.strip():
        write_vector_id(manual_vs.strip())
        st.session_state.vector_store_id = manual_vs.strip()
        st.success(f"Saved Vector Store ID: {manual_vs.strip()}")

    if clear_btn:
        if VECTOR_ID_FILE.exists():
            VECTOR_ID_FILE.unlink()
        st.session_state.vector_store_id = None
        st.info("Cleared saved Vector Store ID.")

    # ----- Create/Reindex flow (GA vector stores) -----
    if create_btn:
        pdf_path = Path(pdf_path_text)
        if not pdf_path.exists():
            st.error(f"File not found: {pdf_path.resolve()}")
            st.stop()

        with st.status("Creating vector store and indexing…", expanded=True) as status:
            try:
                # 1) Create a vector store
                vs = client.vector_stores.create(name="att-10k")
                st.write(f"✓ Vector store created: `{vs.id}`")

                # 2) Upload the file to Files API
                st.write("Uploading PDF to Files API…")
                with open(pdf_path, "rb") as fh:
                    uploaded = client.files.create(file=fh, purpose="assistants")
                st.write(f"✓ File uploaded: `{uploaded.id}` ({getattr(uploaded, 'filename', 'file')})")

                # 3) Attach file to the vector store (starts indexing)
                st.write("Attaching file to vector store (indexing begins)…")
                link = client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded.id)
                st.write(f"✓ Attachment created: `{link.id}`")

                # 4) Poll indexing with timeout
                start = time.time()
                TIMEOUT_S = 8 * 60
                POLL_S = 2
                last_state = None
                state = "unknown"

                while True:
                    fstatus = client.vector_stores.files.retrieve(vector_store_id=vs.id, file_id=uploaded.id)
                    state = getattr(fstatus, "status", getattr(fstatus, "state", "unknown"))
                    if state != last_state:
                        st.write(f"Indexing state: **{state}**")
                        last_state = state

                    if state in ("completed", "failed", "cancelled"):
                        break
                    if time.time() - start > TIMEOUT_S:
                        st.warning("Indexing taking longer than expected; stopping local poll. Server continues in background.")
                        break
                    time.sleep(POLL_S)

                if state == "completed":
                    st.success("Indexing completed.")
                elif state == "failed":
                    err = getattr(fstatus, "last_error", None)
                    st.error(f"Indexing failed. {err or ''}")
                elif state == "cancelled":
                    st.error("Indexing cancelled.")
                else:
                    st.info("Indexing likely still in progress server-side; retrieval will work once complete.")

                # 5) Persist the Vector Store ID
                write_vector_id(vs.id)
                st.session_state.vector_store_id = vs.id
                st.success(f"Vector store saved: `{vs.id}`")
                status.update(label="Create/Reindex finished", state="complete")

            except Exception as e:
                st.exception(e)
                st.stop()

    # ----- Check indexing status -----
    if check_btn:
        vs_id = st.session_state.vector_store_id
        if not vs_id:
            st.warning("No Vector Store ID.")
        else:
            try:
                files = client.vector_stores.files.list(vector_store_id=vs_id, limit=50)
                if not files.data:
                    st.error("Vector store has 0 files attached.")
                else:
                    for f in files.data:
                        fr = client.vector_stores.files.retrieve(vector_store_id=vs_id, file_id=f.id)
                        st.write(f"- file_id: {f.id}  status: **{getattr(fr,'status','unknown')}**")
            except Exception as e:
                st.exception(e)

    # Show current store
    if st.session_state.vector_store_id:
        st.info(f"Active Vector Store:\n`{st.session_state.vector_store_id}`")
    else:
        st.warning("No active vector store. Use Create/Reindex or paste an existing ID.")

st.divider()


# =========================
# Chat history
# =========================
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)


# =========================
# Chat input — Responses API + File Search
# =========================
prompt = st.chat_input("Ask about AT&T’s 10-K (results, segments, debt, risks, notes, etc.)")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        out = st.empty()
        vs_id = st.session_state.vector_store_id

        if not vs_id:
            out.error("No vector store configured yet. Create it in the sidebar or paste an existing ID.")
        else:
            try:
                resp = client.responses.create(
                    model="gpt-4o",  # or "gpt-4o-mini" for lower cost/latency
                    instructions=(
                        "You are a financial-report Q&A bot for AT&T’s 10-K. "
                        "Answer strictly from the document; be concise and numeric where possible. "
                        "Include a brief 'Sources' section with citations to the report. "
                        "If the answer cannot be found, say so."
                    ),
                    input=[{
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }],
                    tools=[{"type": "file_search"}],
                    tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
                )

                # SDK convenience:
                answer = getattr(resp, "output_text", None)
                if not answer:
                    # Robust fallback for older/newer SDK shapes
                    parts = []
                    for outp in getattr(resp, "output", []):
                        for c in getattr(outp, "content", []):
                            if getattr(c, "type", "") in ("output_text", "text"):
                                text_obj = getattr(c, "text", None)
                                parts.append(
                                    getattr(getattr(text_obj, "value", None), "strip", lambda: "")()
                                    if text_obj else getattr(c, "value", "") or ""
                                )
                    answer = "\n".join([p for p in parts if p]) or "_No answer text returned._"

                out.markdown(answer)
                st.session_state.history.append(("assistant", answer))

            except Exception as e:
                out.error(f"OpenAI API error: {e}")
