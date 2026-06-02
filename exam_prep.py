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

# Initialize global question memory bank so it remembers what it asked you previously
if "historical_questions" not in st.session_state:
    st.session_state.historical_questions = []

# Initialize interactive chat history memory bank
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

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
        st.session_state.historical_questions = []  # Reset memory when new files arrive
        st.rerun()

if st.sidebar.button("🗑️ Clear All Uploaded Documents"):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.cache_resource.clear()
    st.session_state.historical_questions = []
    st.session_state.chat_messages = []
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
    "You are an elite Law Exit Examination specialist. Use the provided context to answer questions or build exam papers. "
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
    ["💬 Chat Live with AI", "📝 Interactive Practice Quiz", "⚡ Cram Sheet Engine", "🃏 Flashcard Vault",
     "📊 Global Blueprint Analyzer"]
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
# 💬 CHAT LIVE MODE
# =========================================================
if mode == "💬 Chat Live with AI":
    st.subheader("💬 Interactive Law Study Chat room")
    st.caption("Ask questions, test arguments, or clarify complex concepts based directly on your study materials.")

    # Display past chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept interactive user input chat text box
    if user_chat_input := st.chat_input("Ask anything about your modules or exams..."):
        with st.chat_message("user"):
            st.markdown(user_chat_input)
        st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = rag_chain.invoke({"input": user_chat_input})
                answer = response["answer"]
                st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# =========================================================
# 🚀 INTERACTIVE QUIZ ENGINE (WITH NO-REPEAT ANTI-DUPLICATE MEMORY)
# =========================================================
elif mode == "📝 Interactive Practice Quiz":
    st.subheader("📝 Live Interactive Practice Test")

    num_q = st.slider("Select length of testing blocks:", min_value=5, max_value=30, value=10)

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # Button to clear current block and get fresh questions
    if st.button("🔄 Generate New Test Paper"):
        st.session_state.quiz_questions = None
        st.session_state.submitted = False
        st.rerun()

    # Show historical question count in sidebar for motivation
    st.sidebar.info(
        f"🧠 Memory Bank: AI is remembering {len(st.session_state.historical_questions)} past questions to avoid repeating them!")

    if st.session_state.quiz_questions is None:
        courses_str = ", ".join(selected_courses)

        # Turn the list of historical questions into a string text block for the AI prompt
        forbidden_questions = "\n".join([f"- {q}" for q in
                                         st.session_state.historical_questions]) if st.session_state.historical_questions else "None yet."

        quiz_prompt = f"""
Generate an elite {num_q}-question multiple choice exam based on the following scope: {courses_str}.
Ensure the questions strictly mimic real Law Exit Exam questions.

CRITICAL ANTI-REPETITION CONSTRAINT:
Do NOT generate any questions similar or identical to these previously generated questions. You must create completely fresh problems:
{forbidden_questions}

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
        with st.spinner("🤖 Simulating non-repeated test parameters and tracking citations..."):
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
                    citation = re.search(r"Source Citation:\s*(.*)", block).group(1).strip()

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

                    # Add question to permanent history session state memory bank so it is locked out next run
                    if q_text not in st.session_state.historical_questions:
                        st.session_state.historical_questions.append(q_text)
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

        with st.form(key="quiz_evaluation_form"):
            temp_answers = {}
            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"#### Question {idx + 1}: {q['question']}")
                options_keys = list(q.get('options', {}).keys())

                user_choice = st.radio(
                    f"Choose option for Q{idx + 1}:",
                    options_keys,
                    format_func=lambda x: q.get('options', {}).get(x, ''),
                    key=f"q_radio_{idx}"
                )
                temp_answers[idx] = user_choice
                st.write("---")

            submit_clicked = st.form_submit_button("🎯 Submit Answers & Calculate Final Grade")

            if submit_clicked:
                st.session_state.submitted = True
                st.session_state.saved_answers = temp_answers

        if st.session_state.submitted and "saved_answers" in st.session_state:
            score = 0
            total = len(st.session_state.quiz_questions)

            st.markdown("### 📊 Your Results Breakdown")
            for idx, q in enumerate(st.session_state.quiz_questions):
                user_ans = st.session_state.saved_answers.get(idx)
                is_correct = str(user_ans).strip().startswith(q['correct'])

                if is_correct:
                    score += 1
                    st.success(f"Question {idx + 1}: Correct! Your Answer: {user_ans}")
                else:
                    st.error(f"Question {idx + 1}: Wrong. Your Answer: {user_ans} | Correct Answer: {q['correct']}")

                st.markdown(f"🔖 NotebookLM Source Reference: {q['citation']}")
                st.markdown("🔍 Choices Breakdown Analysis:")
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
# OTHER MODES
# =========================================================
elif mode == "⚡ Cram Sheet Engine":
    st.subheader("⚡ High-Density Cram Sheet Generator")
    courses_str = ", ".join(selected_courses)
    st.write(f"Targeting Curriculum Area: {courses_str}")

    if st.button("🚀 Generate Revision Guides"):
        with st.spinner("🤖 Extracting core structures..."):
            cram_prompt = f"Create a comprehensive, bulleted high-density cram study guide summarizing key concepts, definitions, and rules for these specific subjects: {courses_str}."
            response = rag_chain.invoke({"input": cram_prompt})
            st.markdown("### 📝 Generated Cram Sheet Guide")
            st.markdown(response["answer"])

elif mode == "🃏 Flashcard Vault":
    st.subheader("🃏 Smart Flashcard Vault (Organized by Course)")
    courses_str = ", ".join(selected_courses)
    st.write(f"Currently targeting: {courses_str}")

    if st.button("🚀 Build Digital Flashcards"):
        with st.spinner("🤖 Designing modular study flashcards..."):
            flash_prompt = f"Create 15 concise, highly effective study flashcards covering the active scope: {courses_str}. Separate the cards cleanly by individual course names, focusing strictly on definitions, case rules, or provisions. Use standard Q: and A: presentation blocks."
            response = rag_chain.invoke({"input": flash_prompt})
            st.markdown("### 🃏 Generated Flashcards Vault")
            st.markdown(response["answer"])

elif mode == "📊 Global Blueprint Analyzer":
    st.subheader("📊 Curriculum vs Past Paper Blueprint Matrix")
    courses_str = ", ".join(selected_courses)
    st.markdown(
        "This strategic analyzer cross-references your 17-course exit exam curriculum requirements "
        "against the legal core concepts inside your vector knowledge base to reveal high-probability "
        "topics, concept weightings, and key structural trends."
    )

    if st.button("🚀 Run Comprehensive Cross-Match Analysis"):
        with st.spinner("🤖 Mapping curriculum structures to past exam papers..."):
            blueprint_prompt = f"""You are an expert academic psychometrician and law exit examination strategist.
            Analyze the core legal knowledge base context provided to reverse-engineer a comprehensive Exam Blueprint Matrix.

            Please synthesize and display a detailed report containing:
            1. Core Concept Weight Analysis: Scan the database for topics related to these active courses: {courses_str}. Identify which specific subjects, doctrines, or legal tests have the highest concentration of details or deep explanations.
            2. High-Probability Question Focus Areas: Outline specific areas (e.g., 'Formation of Contracts' in Contract Law or 'Homicide Elements' in Criminal Law) that are absolutely critical for a final 100-question exit examination.
            3. Strategic Study Plan Table: Build a Markdown-formatted table mapping out the active subjects, their estimated topic weights (High/Medium/Low based on details density), and crucial statutory sections or case principles students must memorize.

            Ensure the output uses clean Markdown tables, bold headers, and structured bullet points."""

            response = rag_chain.invoke({"input": blueprint_prompt})
            st.markdown("### 🗺️ Generated Curriculum Blueprint Matrix")
            st.markdown(response["answer"])
