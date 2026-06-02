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

if uploaded_files:
    new_file_added = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"✅ Saved: {uploaded_file.name}")
            new_file_added = True

    if new_file_added:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.cache_resource.clear()
        st.rerun()

if st.sidebar.button("🗑️ Clear All Uploaded Documents"):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.cache_resource.clear()
    st.sidebar.warning("🧹 Cleared all documents!")
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
    "If the answer cannot be found in the context, synthesize the most legally sound response.\n\n{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# =========================================================
# 🎛️ SIDEBAR NAVIGATION
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
# 🚀 INTERACTIVE QUIZ ENGINE
# =========================================================
if mode == "Interactive Practice Quiz":
    st.subheader("📝 Live Interactive Practice Test")

    num_q = st.slider("Select length of testing blocks:", min_value=5, max_value=30, value=10)

    # Initialize Quiz Session variables safely across user interaction refreshes
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if st.button("🔄 Generate New Test Paper"):
        st.session_state.quiz_questions = None
        st.session_state.submitted = False
        st.rerun()

    if st.session_state.quiz_questions is None:
        courses_str = ", ".join(selected_courses)
        quiz_prompt = f"""
    Generate an elite {num_q}-question multiple choice exam based on the following scope: {courses_str}.
    Ensure the questions strictly mimic real Law Exit Exam questions.

    CRITICAL INSTRUCTIONS:
    1. You must track which specific source document or past paper you used to extract this question.
    2. You must provide a clear, dedicated explanation for EVERY single multiple choice option, outlining why that specific choice is correct or incorrect.

    You MUST format the output EXACTLY like the following example format so my custom parser can read it seamlessly. Do not deviate from this layout structure:

    Q1: What is the primary definition of a contract?
    A) An agreement enforceable by law
    B) A casual verbal promise
    C) A social arrangement
    D) A unilateral non-binding declaration
    Correct Answer: A
    Source Citation: [From: Contract Law Module 2024 / Past Exam 2022 Q14]
    Option A Explanation: (Correct) Under contract law, a contract requires legal intent and mutual enforcement mechanisms.
    Option B Explanation: (Incorrect) A casual verbal promise lacks legal intent and formal consideration.
    Option C Explanation: (Incorrect) Social arrangements are generally presumed to lack an intention to create legal relations.
    Option D Explanation: (Incorrect) A unilateral declaration is missing the necessary element of mutual agreement.

    Q2: [Next Question Here]
    ...
    """
        with st.spinner("🤖 Simulating test parameters, tracking citations, and building options breakdowns..."):
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
                        letter = o[0]
                        content = o[2:].strip()
                        options_dict[letter] = f"{letter}) {content}"

                    correct = re.search(r"Correct Answer:\s*([A-D])", block).group(1).strip()
                    citation = re.search(r"Source Citation:\s*(.*)", block).group(1).strip()

                    # Group all custom options explanations together
                    exp_block = ""
                    opt_exps = re.findall(r"(Option [A-D] Explanation:.*)", block)
                    if opt_exps:
                        exp_block = "\n".join([f"* {e}" for e in opt_exps])
                    else:
                        exp_block = re.search(r"Explanation:\s*(.*)", block, re.DOTALL).group(1).strip()

                    questions_data.append({
                        "question": q_text,
                        "options": options_dict,
                        "correct": correct,
                        "citation": citation,
                        "explanation": exp_block
                    })
                except Exception:
                    continue

            if questions_data:
                st.session_state.quiz_questions = questions_data
            else:
                st.session_state.quiz_questions = "FALLBACK"
                st.session_state.fallback_text = raw_text

    # Render out the stable Interactive Form Engine
    if st.session_state.quiz_questions == "FALLBACK":
        st.warning("⚠️ High layout complexity detected. Rendering raw structured sheet version below:")
        st.write(st.session_state.fallback_text)

    elif st.session_state.quiz_questions:
        st.info(f"📋 Running Live Simulated Evaluation across target courses.")

        # WE PACK ALL QUESTIONS INSIDE A FIXED FORM TO PREVENT INSTANT VANISHING REFRESHEINGS!
        with st.form(key="quiz_evaluation_form"):
            temp_answers = {}
            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"#### **Question {idx + 1}:** {q['question']}")

                options_keys = list(q.get('options', {}).keys())

                user_choice = st.radio(
                    f"Choose option for Q{idx + 1}:",
                    options_keys,
                    format_func=lambda x: q.get('options', {}).get(x, ''),
                    key=f"q_radio_{idx}"
                )
                temp_answers[idx] = user_choice
                st.write("---")

            # Form actions button
            submit_clicked = st.form_submit_button("🎯 Submit Answers & Calculate Final Grade")

            if submit_clicked:
                st.session_state.submitted = True
                st.session_state.saved_answers = temp_answers

        # Outside the form container, show the calculations results breakdown permanently
        if st.session_state.submitted and "saved_answers" in st.session_state:
            score = 0
            total = len(st.session_state.quiz_questions)

            st.markdown("### 📊 Your Results Breakdown")
            for idx, q in enumerate(st.session_state.quiz_questions):
                user_ans = st.session_state.saved_answers.get(idx)
                is_correct = user_ans == q['correct']

                if is_correct:
                    score += 1
                    st.success(f"**Question {idx + 1}:** Correct! Your Answer: {user_ans}")
                else:
                    st.error(f"**Question {idx + 1}:** Wrong. Your Answer: {user_ans} | Correct Answer: {q['correct']}")

                # Show source citation and option breakdown underneath
                st.markdown(f"🔖 **NotebookLM Source Reference:** `{q['citation']}`")
                st.markdown("**🔍 Choices Breakdown Analysis:**")
                st.markdown(q['explanation'])
                st.write("---")

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
        st.write(f"Targeting Curriculum Area: `{courses_str}`")
        if st.button("🚀 Generate Revision Guides"):
            with st.spinner("🤖 Extracting core structures..."):
                prompt = f"Create a comprehensive, bulleted high-density cram study guide summarizing key concepts, definitions, and rules for these specific subjects: {courses_str}."
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])


    elif mode == "Flashcard Vault":

        st.subheader("🃏 Smart Flashcard Vault (Organized by Course)")

        courses_str = ", ".join(selected_courses)

        st.write(f"Currently targeting: `{courses_str}`")

        if st.button("🚀 Build Digital Flashcards"):
            with st.spinner("🤖 Designing modular study flashcards..."):
                flash_prompt = f"""

    Create 15 concise, highly effective study flashcards covering the active scope: {courses_str}.


    REQUIREMENTS:

    1. Separate and group the flashcards cleanly by their individual course names.

    2. Focus strictly on definitions, case rules, statutory provisions, or key operational concepts relevant to law exit exams.

    3. Use a clear question-and-answer presentation layout.


    Format exactly like this:

    ### 📚 [Course Name]

    * **Flashcard 1**

      * **Q:** What is the legal definition of [Concept]?

      * **A:** [Answer based on study materials]

    """

                response = rag_chain.invoke({"input": flash_prompt})

                st.markdown(response["answer"])



    elif mode == "Global Blueprint Analyzer":

        st.subheader("📊 Curriculum vs Past Paper Blueprint Matrix")

        st.markdown(

            "This analyzer cross-references your 17-course exit exam curriculum outline against all "

            "uploaded past exam documents to reveal exactly where the questions are being extracted from, "

            "historical question volumes per chapter, and weight trends."

        )

        if st.button("🚀 Run Comprehensive Cross-Match Analysis"):
            with st.spinner("🤖 Mapping curriculum structures to past exam papers..."):
                blueprint_prompt = """

    Analyze all local files in the study database to trace exam weights and distributions.


    Please provide a highly structured, data-driven report outlining:

    1. **Course & Chapter Blueprint Distribution**: Grouped by the 17 curriculum subjects, map out which specific chapters or sub-topics appear most frequently in the past papers.

    2. **Question Volumes**: Estimate how many questions traditionally originate from each identified chapter block based on the past exams provided.

    3. **Document Tracking & Source References**: For every major chapter weight trend identified, explicitly cite the specific past exam file names, question numbers, or module files where you found the overlapping concepts.


    Format the output clearly using Markdown tables, headers, and bullet points so it reads like a professional analytical report.

    """

                response = rag_chain.invoke({"input": blueprint_prompt})

                st.markdown(response["answer"])

