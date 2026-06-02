import os
import re
import shutil
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# ⚙️ APP CONFIGURATION
# =========================================================
st.set_page_config(page_title="Elite Law Exit Exam Prep", page_icon="⚖️", layout="wide")
st.title("⚖️ Elite Law Exit Exam Prep Dashboard")
st.subheader("Master your 17 curriculum courses and crush the final 100-question exam!")

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Missing Groq API Key! Please configure your secrets panel in Streamlit Advanced Settings.")
    st.stop()

# Ensure local directories exist
UPLOAD_DIR = "study_material"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# 📂 DRAG & DROP FILE UPLOADER WIDGET
# =========================================================
st.sidebar.header("📤 Study Material Dropzone")
uploaded_files = st.sidebar.file_uploader(
    "Upload new PDFs or Text files for you and your friends:",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Process newly uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
            # Clear old cached vector index so it rebuilds with new documents included
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.cache_resource.clear()

# Reset button if you want to wipe uploaded files clean
if st.sidebar.button("🗑️ Clear All Uploaded Documents"):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.cache_resource.clear()
    st.sidebar.warning("🧹 Cleared all documents. Upload new ones to begin!")
    st.rerun()

# =========================================================
# 📚 CURRICULUM DATA DIRECTORY
# =========================================================
CURRICULUM = {
    "Private Law 📜": ["Family Law", "Property Law", "Succession Law", "Commercial Law", "Contract Law",
                      "Extra-contractual Liability (Tort)"],
    "Public Law 🏛️": ["Constitutional Law", "Criminal Law", "Human Rights Law", "Public International Law", "Tax Law"],
    "Miscellaneous 💼": ["Employment Law", "Jurisprudence", "Legal Ethics"],
    "Skills 🛠️": ["Criminal Procedure", "Civil Procedure", "Law of Evidence"]
}
ALL_17_COURSES = [course for sublist in CURRICULUM.values() for course in sublist]


# =========================================================
# 🧠 CACHED VECTOR STORE ENGINE
# =========================================================
@st.cache_resource
def initialize_vector_store():
    documents = []
    for file_path in glob.glob(f"{UPLOAD_DIR}/**/*.*", recursive=True):
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
        except Exception:
            pass

    if not documents:
        return None

    # Optimized chunking for law materials
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store


with st.spinner("🧠 Analyzing all 17 course materials & past papers... Standby..."):
    vector_store = initialize_vector_store()

if vector_store is None:
    st.info("📁 The knowledge base is currently empty. Use the sidebar drag-and-drop zone to upload your law documents!")
    st.stop()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.4)

system_prompt = (
    "You are an elite Law Exit Examination specialist. Use the provided context to build accurate exam questions. "
    "If the answer cannot be found in the context, synthesize the most legally sound response based on standard legal principles.\n\n"
    "{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# =========================================================
# 🎛️ SIDEBAR NAVIGATION WITH CURRICULUM CHIPS
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("🎯 Navigation Panel")
mode = st.sidebar.radio(
    "Select System Mode:",
    ["Interactive Practice Quiz", "Cram Sheet Engine", "Flashcard Vault", "Global Blueprint Analyzer"]
)

st.sidebar.markdown("---")
st.sidebar.header("📚 Course Selection")

exam_type = st.sidebar.selectbox(
    "Choose Target Scope:",
    ["Simulated Final Exam (Mix of all 17 Courses)", "Targeted Categories / Single Course"]
)

selected_courses = []
if exam_type == "Simulated Final Exam (Mix of all 17 Courses)":
    selected_courses = ALL_17_COURSES
    st.sidebar.success("🔗 Mode: Comprehensive 17-Course Mix enabled!")
else:
    category = st.sidebar.selectbox("Select Law Category:", list(CURRICULUM.keys()))
    specific_course = st.sidebar.selectbox("Select Specific Subject:", ["All in this Category"] + CURRICULUM[category])
    if specific_course == "All in this Category":
        selected_courses = CURRICULUM[category]
    else:
        selected_courses = [specific_course]

# =========================================================
# 🚀 INTERACTIVE QUIZ ENGINE (WITH REAL-TIME SCORING)
# =========================================================
if mode == "Interactive Practice Quiz":
    st.subheader("📝 Live Interactive Practice Test")

    num_q = st.slider("Select length of testing blocks:", min_value=5, max_value=30, value=10)

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if st.button("🔄 Generate New Test Paper"):
        st.session_state.quiz_questions = None
        st.session_state.user_answers = {}
        st.session_state.submitted = False
        st.rerun()

    if st.session_state.quiz_questions is None:
        courses_str = ", ".join(selected_courses)
        quiz_prompt = f"""
Generate an elite {num_q}-question multiple choice exam based on the following scope: {courses_str}.
Ensure the questions strictly mimic real Law Exit Exam questions.

You MUST format the output EXACTLY like the following example format so my parser can process it seamlessly. Do not deviate:

Q1: What is the primary definition of a contract?
A) An agreement enforceable by law
B) A casual verbal promise
C) A social arrangement
D) A unilateral non-binding declaration
Correct Answer: A
Explanation: A contract requires legal intent and mutual enforcement mechanisms under contract law.

Q2: [Next Question Here]
...
"""
        with st.spinner("🤖 Simulating test parameters and extraction..."):
            response = rag_chain.invoke({"input": quiz_prompt})
            raw_text = response["answer"]

            parsed_blocks = re.findall(r"(Q\d+:.*?)(?=Q\d+:|$)", raw_text, re.DOTALL)

            questions_data = []
            for block in parsed_blocks:
                try:
                    q_text = re.search(r"Q\d+:(.*?)(?=[A-D]\))", block, re.DOTALL).group(1).strip()
                    opts = re.findall(r"([A-D]\).*?)(?=[A-D]\)|Correct Answer:|$)", block, re.DOTALL)
                    options_dict = {}
                    for o in opts:
                        letter = o
                        content = o[2:].strip()
                        options_dict[letter] = f"{letter}) {content}"

                    correct = re.search(r"Correct Answer:\s*([A-D])", block).group(1).strip()
                    explanation = re.search(r"Explanation:\s*(.*)", block, re.DOTALL).group(1).strip()

                    questions_data.append({
                        "question": q_text,
                        "options": options_dict,
                        "correct": correct,
                        "explanation": explanation
                    })
                except Exception:
                    continue

            if questions_data:
                st.session_state.quiz_questions = questions_data
            else:
                st.session_state.quiz_questions = "FALLBACK"
                st.session_state.fallback_text = raw_text

        if st.session_state.quiz_questions == "FALLBACK":
            st.warning("⚠️ High layout complexity detected. Rendering raw structured sheet version below:")
            st.write(st.session_state.fallback_text)

        elif st.session_state.quiz_questions:
            st.info(f"📋 Running Live Simulated Evaluation across target courses.")

            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"#### Question {idx + 1}: {q['question']}")

                options_keys = list(q.get('options', {}).keys())
                current_choice = st.session_state.user_answers.get(idx, None)
                default_idx = options_keys.index(current_choice) if current_choice in options_keys else 0

                user_choice = st.radio(
                    f"Choose option for Q{idx + 1}:",
                    options_keys,
                    format_func=lambda x: q.get('options', {}).get(x, ''),
                    index=default_idx,
                    key=f"q_radio_{idx}"
                )
                st.session_state.user_answers[idx] = user_choice
                st.write("---")

            if not st.session_state.submitted:
                if st.button("🎯 Submit Answers & Calculate Final Grade"):
                    st.session_state.submitted = True
                    st.rerun()

            if st.session_state.submitted:
                score = 0
                total = len(st.session_state.quiz_questions)

                st.markdown("### 📊 Your Results Breakdown")
                for idx, q in enumerate(st.session_state.quiz_questions):
                    user_ans = st.session_state.user_answers.get(idx)
                    is_correct = user_ans == q['correct']

                    if is_correct:
                        score += 1
                        st.success(f"Question {idx + 1}: Correct! Your Answer: {user_ans}")
                    else:
                        st.error(f"Question {idx + 1}: Wrong. Your Answer: {user_ans} | Correct Answer: {q['correct']}")
                    st.caption(f"💡 Explanation: {q['explanation']}")

                percentage = (score / total) * 100
                st.metric("🏆 Final Test Score", f"{score} / {total}", f"{percentage:.1f}% Match Rate")

                if percentage >= 70:
                    st.balloons()
                    st.success("🔥 Elite status achieved! You and your friends are ready to ace this section!")
                else:
                    st.warning("📚 Solid attempt! Focus on the text summaries and rerun another simulated block.")

        # =========================================================
        # OTHER MODES (CRAM SHEET, FLASHCARDS, ETC.)
        # =========================================================
    elif mode == "Cram Sheet Engine":
        st.subheader("⚡ High-Density Cram Sheet Generator")
        courses_str = ", ".join(selected_courses)
        st.write(f"Targeting Curriculum Area: {courses_str}")

        if st.button("🚀 Generate Revision Guides"):
            with st.spinner("🤖 Extracting core structures..."):
                prompt = f"Create a comprehensive, bulleted high-density cram study guide summarizing key concepts, definitions, and rules for these specific subjects: {courses_str}."
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])

    elif mode == "Flashcard Vault":
        st.subheader("🃏 Smart Flashcard Review")
        courses_str = ", ".join(selected_courses)

        if st.button("🚀 Build Digital Flashcards"):
            with st.spinner("🤖 Designing flashcards..."):
                prompt = f"Create 15 concise flashcards for the selected modules: {courses_str}. Format clearly with Q: and A: blocks."
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])

    elif mode == "Global Blueprint Analyzer":
        st.subheader("📊 Curriculum vs Past Paper Gap Analysis")

        if st.button("🚀 Run Comprehensive Cross-Match Analysis"):
            with st.spinner("🤖 Running global analysis across curriculum structure..."):
                prompt = "Cross-analyze all local files in the study database. Outline overlapping core definitions, recurring exam questions across years, and classify priority study points."
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
