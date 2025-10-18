# policy_explainer_assistant.py
import streamlit as st
import os
import tempfile
import json
from typing import List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_groq import ChatGroq

# Groq API
from groq import Groq

# Document processing
import docx2txt


# -------------------- CONFIGURATION --------------------
class Config:
    """
    Central configuration class with your exact settings
    """

    def __init__(self):
        # Your exact API key and settings
        self.GROQ_API_KEY = "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"
        self.MODEL_NAME = "llama-3.1-8b-instant"
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.SIMILARITY_SEARCH_K = 4
        self.MAX_CHAT_HISTORY = 20
        self.TEMPERATURE = 0.1
        self.APP_MODE = "policy"

        # Vector store settings
        self.VECTOR_STORE_PATH = "./vector_cache"

        # App settings
        self.APP_NAME = "Policy Explainer & Study Assistant"

    def validate_config(self):
        """Validate that all required configurations are set"""
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")
        return True


config = Config()

# -------------------- CUSTOM CSS --------------------
CUSTOM_CSS = """
<style>
    body, .stApp {
        background-color: #fdfdfd;
        color: #222;
        font-size: 0.9rem;
    }

    h1, h2, h3, h4, h5 {
        color: #1a1a1a;
        font-weight: 600;
    }

    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 10px;
        font-size: 0.85rem;
        padding: 0.4rem 0.9rem;
        border: none;
    }

    .stButton > button:hover {
        background-color: #2563eb !important;
        color: #fff !important;
    }

    .main-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #eaeaea;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .notes-container {
        background-color: #fff9eb;
        padding: 2rem;
        border-radius: 14px;
        border: 2px solid #ffd54f;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .bullet-points {
        background-color: transparent;
        padding: 0;
        margin: 0;
    }
    
    .bullet-points ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .bullet-points li {
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .quiz-box {
        background: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }
    
    /* Custom button styles */
    .quiz-button {
        background-color: #ef4444 !important;
        color: white !important;
    }
    
    .quiz-button:hover {
        background-color: #dc2626 !important;
        color: white !important;
    }
    
    .secondary-button {
        background-color: #6b7280 !important;
        color: white !important;
    }
    
    .secondary-button:hover {
        background-color: #4b5563 !important;
        color: white !important;
    }
    
    .stRadio label {
        color: #1a1a1a !important;
    }
    
    .stRadio > div > label > div:last-child {
        color: #1a1a1a !important;
    }
</style>
"""


# -------------------- DOCUMENT PROCESSOR --------------------
class DocumentProcessor:
    """
    Handles document loading, text cleaning, and chunking for retrieval.
    Works for both Policy and Student document modes.
    """

    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    def load_document(self, file_path: str) -> str:
        """Loads a document (PDF, DOCX, TXT) and returns cleaned text."""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text = "\n".join([d.page_content for d in docs])
            elif ext == ".docx":
                text = docx2txt.process(file_path)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            return self._clean_text(text)

        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing redundant spaces, page numbers, and formatting issues."""
        import re

        # Remove multiple newlines, page numbers, and unnecessary spacing
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"Page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip()

        return text

    def chunk_document(self, text: str, metadata: dict = None):
        """
        Splits long text into manageable chunks for embedding and retrieval.
        Returns list of LangChain Document objects.
        """
        if not text.strip():
            raise ValueError("Empty document text. Cannot create chunks.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

        chunks = text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks
        ]

        return documents


# -------------------- VECTOR STORE MANAGER --------------------
class VectorStoreManager:
    """
    Manages the creation and handling of vector stores using FAISS and HuggingFace embeddings.
    """

    def __init__(self):
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
            # Removed print statement to avoid console output in Streamlit
        except Exception as e:
            raise RuntimeError(f"Error initializing embedding model: {str(e)}")

    def create_vector_store(self, documents):
        """
        Converts processed document chunks into FAISS vector embeddings.
        """
        if not documents or len(documents) == 0:
            raise ValueError("Cannot create vector store from an empty document list.")

        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            # Removed print statement - success will be shown in the UI
            return vector_store

        except Exception as e:
            raise RuntimeError(f"Error creating vector store: {str(e)}")


# -------------------- SUMMARIZER --------------------
class PolicySummarizer:
    """
    Handles summarization of policy or study documents.
    Uses Groq's Llama-3 model to generate clear, structured summaries.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=config.TEMPERATURE,
        )

    def summarize_document(self, document_text: str) -> str:
        """
        Generates a simple and structured summary of the given document text.
        """
        if not document_text or not document_text.strip():
            return "⚠️ No document text available to summarize."

        # Limit input length to prevent model overload
        text = document_text[:12000]

        template = """
        You are an AI assistant that simplifies complex text into clear, concise summaries.

        Summarize the following document in easy-to-understand language.
        Highlight the key ideas, important rules, and any critical details.

        Make sure your summary includes:
        - ✅ The main purpose or topic
        - 📘 Key rules, points, or sections
        - 🧠 Important takeaways or principles
        - 📄 Any exceptions or special cases
        - 💡 Overall summary in simple language

        If this is a study or textbook material, also:
        - Provide concise notes or bullet points for each concept
        - Simplify explanations to help students understand easily

        Use bullet points and short paragraphs for readability.

        --- Document ---
        {document_text}
        """

        prompt = PromptTemplate(
            input_variables=["document_text"], template=template.strip()
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            summary = chain.run(document_text=text)
            return summary.strip()
        except Exception as e:
            return f"⚠️ Error while summarizing: {str(e)}"


# -------------------- Q&A SYSTEM --------------------
class PolicyQASystem:
    """
    Handles question answering using the Groq LLM + vector retrieval.
    Supports both policy document queries and study material clarifications.
    """

    def __init__(self, vector_store):
        if vector_store is None:
            raise ValueError("Vector store is not initialized.")

        self.llm = ChatGroq(
            model=config.MODEL_NAME,
            groq_api_key=config.GROQ_API_KEY,
            temperature=config.TEMPERATURE,
        )

        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": config.SIMILARITY_SEARCH_K}
        )

        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        template = """
        You are a helpful assistant that answers questions based on provided context.
        Keep your answer clear, concise, and beginner-friendly.

        Context:
        {context}

        Question:
        {question}

        Provide your answer in a way that's:
        - Simple and direct
        - Step-by-step if needed
        - Avoids jargon
        - Helpful for understanding the main point
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template.strip(),
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        return qa_chain

    def ask_question(self, question: str):
        """Handles user query and returns LLM-generated answer."""
        if not question.strip():
            return {"answer": "Please ask a valid question.", "source_documents": []}

        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
            }
        except Exception as e:
            return {
                "answer": f"⚠️ Error while answering: {str(e)}",
                "source_documents": [],
            }


# -------------------- FLASHCARD GENERATOR --------------------
class StudyFlashcardGenerator:
    """
    Generates concise flashcard-style cues for revision based on a study summary.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=config.MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            temperature=config.TEMPERATURE,
        )

        template = """
You are a concise flashcard generator.

From the given study summary, extract 8–12 key concepts or cues for quick revision.

Follow these rules strictly:
- Output each item as a separate Markdown bullet point on a new line.
- Each bullet must follow this format: - **Topic Name:** short explanation.
- Keep explanations brief (maximum 12–15 words).
- Do not include any introductory or closing sentences.
- Do not include numbering, section titles, or labels like "Flashcards" or "Concepts".
- Focus on clarity and usefulness for quick recall.

Example output:
- **Photosynthesis:** Process where plants convert sunlight into energy.
- **Newton's Laws:** Describe the relationship between motion and forces.
- **Supply and Demand:** Explain balance between market needs and product availability.

Now, create the cues based on this summary:

{summary_text}
"""
        self.prompt = PromptTemplate(
            input_variables=["summary_text"], template=template.strip()
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_flashcards(self, summary_text: str) -> str:
        """
        Generate flashcards (markdown bullets) from the provided summary_text.
        """
        if not summary_text or not summary_text.strip():
            return "⚠️ No summary available to generate flashcards."

        limited_text = summary_text[:10000]

        try:
            output = self.chain.run(summary_text=limited_text)
            return output.strip()
        except Exception as e:
            return f"⚠️ Error generating flashcards: {str(e)}"


# -------------------- QUIZ GENERATOR --------------------
class QuizGenerator:
    """
    Uses Groq LLM to generate a structured quiz (MCQs)
    from summarized study material.
    """

    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        # Use a working model - updated to current Groq models
        self.model = "llama-3.1-8b-instant"  # Using the same model as other components

    def _generate_quiz_prompt(self, study_summary: str) -> str:
        return f"""
You are a helpful study quiz generator.

Given the following summarized study material, create **exactly 5 multiple-choice questions** (MCQs)
that test conceptual understanding.

Each question should:
- Have a **clear and concise question text**.
- Include **4 options (A, B, C, D)**.
- Indicate the **correct answer** (A–D).
- Be suitable for a student revising the material.
- Avoid repeating questions.

Return your output **only** as valid JSON with this structure:

{{
  "quiz": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "correct_answer": "A"
    }}
  ]
}}

Do not include any explanations, notes, or markdown — only JSON.

STUDY MATERIAL SUMMARY:
\"\"\"
{study_summary}
\"\"\"
"""

    def generate_quiz(self, study_summary: str) -> List[Dict[str, Any]]:
        """
        Calls the Groq API to generate and parse the quiz.
        """
        try:
            if not study_summary or not study_summary.strip():
                st.error("No study summary available to generate quiz.")
                return []

            # Limit the summary length to avoid token limits
            limited_summary = study_summary[:8000]

            prompt = self._generate_quiz_prompt(limited_summary)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise quiz generator that outputs only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            raw_output = response.choices[0].message.content.strip()

            # Clean the output and extract JSON
            cleaned_output = raw_output.strip()

            # Remove any markdown code block markers
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]
            elif cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[3:]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]

            cleaned_output = cleaned_output.strip()

            # Attempt to locate JSON portion only
            start_idx = cleaned_output.find("{")
            end_idx = cleaned_output.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                st.error(
                    "Failed to generate valid quiz format. The model didn't return proper JSON."
                )
                st.info("Raw response for debugging:")
                st.code(cleaned_output)
                return []

            json_str = cleaned_output[start_idx:end_idx]

            quiz_data = json.loads(json_str)
            quiz_items = quiz_data.get("quiz", [])

            # Validate the quiz data structure
            if not quiz_items or not isinstance(quiz_items, list):
                st.error("Generated quiz format is invalid. Please try again.")
                return []

            # Validate each question has required fields
            valid_questions = []
            for i, question in enumerate(quiz_items):
                if (
                    isinstance(question, dict)
                    and "question" in question
                    and "options" in question
                    and "correct_answer" in question
                ):
                    valid_questions.append(question)
                else:
                    st.warning(f"Question {i+1} has missing fields and was skipped")

            if not valid_questions:
                st.error("No valid questions were generated. Please try again.")
                return []

            return valid_questions

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse quiz JSON: {str(e)}")
            st.info("Raw response for debugging:")
            st.code(raw_output if "raw_output" in locals() else "No response")
            return []
        except Exception as e:
            st.error(f"Quiz generation error: {str(e)}")
            return []


# -------------------- QUICK NOTES PAGE --------------------
def quick_notes_page():
    """Quick Notes page functionality"""
    st.set_page_config(
        page_title="🧠 Quick Notes",
        page_icon="🧠",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🧠 Quick Notes")
    st.markdown(
        "These are concise key points from your study material — perfect for last-minute revision!"
    )
    st.markdown("---")

    if "flashcard_data" in st.session_state and st.session_state.flashcard_data:
        notes_raw = st.session_state.flashcard_data.strip()

        # Convert text to proper bullet list formatting
        notes_list = []
        for line in notes_raw.splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                line = line.lstrip("-• ").strip()
            if line:
                notes_list.append(line)

        # Create proper bullet points with better formatting
        formatted_notes = ""
        for item in notes_list:
            if ":" in item:
                # Split by first colon to separate topic from explanation
                parts = item.split(":", 1)
                topic = parts[0].strip()
                explanation = parts[1].strip() if len(parts) > 1 else ""
                formatted_notes += f"- **{topic}:** {explanation}\n"
            else:
                formatted_notes += f"- {item}\n"

        st.markdown(formatted_notes)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning(
            "⚠️ No Quick Notes found. Please generate notes from the Study Assistant tab first."
        )

    st.markdown("---")

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "⬅ Back to Study Assistant", key="back_from_notes", use_container_width=True
        ):
            st.session_state.current_page = "main"
            st.rerun()

    with col2:
        if st.button(
            "🎯 Take Quiz",
            key="take_quiz_from_notes",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.current_page = "quiz"
            st.rerun()


# -------------------- QUIZ PAGE --------------------
def quiz_page():
    """Quiz page functionality"""
    st.set_page_config(page_title="🧠 Study Quiz", page_icon="🧩", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🧠 Study Material Quiz")
    st.markdown("Answer multiple-choice questions generated from your study material!")

    # Session check
    if "study_summary" not in st.session_state or not st.session_state.study_summary:
        st.warning("⚠️ Please upload study material in the Fresher tab first.")
        if st.button("⬅ Back to Main App"):
            st.session_state.current_page = "main"
            st.rerun()
        return

    # Quiz generation - always regenerate to ensure fresh quiz
    if "quiz_data" not in st.session_state or st.session_state.quiz_data is None:
        with st.spinner("Generating quiz questions... ⏳"):
            quiz_gen = QuizGenerator()
            try:
                quiz_data = quiz_gen.generate_quiz(st.session_state.study_summary)
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.success(
                        f"✅ Quiz generated successfully! {len(quiz_data)} questions created."
                    )
                else:
                    st.error("❌ Failed to generate quiz. Please try again.")
                    if st.button("🔄 Retry Quiz Generation"):
                        del st.session_state.quiz_data
                        st.rerun()
                    return
            except Exception as e:
                st.error(f"❌ Error generating quiz: {str(e)}")
                if st.button("🔄 Retry Quiz Generation"):
                    del st.session_state.quiz_data
                    st.rerun()
                return

    quiz_data = st.session_state.quiz_data

    # Check if quiz_data is valid
    if not quiz_data or not isinstance(quiz_data, list):
        st.error(
            "❌ No valid quiz data available. Please try generating the quiz again."
        )
        if st.button("🔄 Regenerate Quiz"):
            del st.session_state.quiz_data
            st.rerun()
        return

    st.markdown("---")
    st.subheader(f"📋 Quiz Questions ({len(quiz_data)} questions)")

    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(quiz_data, 1):
            st.markdown(f"#### Q{i}. {q['question']}")
            options = q.get("options", {})
            if not options:
                st.error(f"Question {i} has no options")
                continue

            option_values = list(options.values())
            option_keys = list(options.keys())

            # Display options with labels
            user_choice = st.radio(
                label=f"Select an answer for Q{i}:",
                options=option_values,
                key=f"q{i}",
                index=None,
                label_visibility="collapsed",
            )

            if user_choice:
                # Find the key for the chosen option
                chosen_index = option_values.index(user_choice)
                user_answers[i] = option_keys[chosen_index]
            else:
                user_answers[i] = None

            st.markdown("")

        submitted = st.form_submit_button("✅ Submit Answers", use_container_width=True)

    # Result evaluation
    if submitted:
        correct = 0
        total = len(quiz_data)
        results = []

        for i, q in enumerate(quiz_data, 1):
            correct_answer = q.get("correct_answer", "").strip().upper()
            user_answer = user_answers.get(i)

            is_correct = user_answer == correct_answer if user_answer else False
            if is_correct:
                correct += 1

            results.append(
                {
                    "question_num": i,
                    "question": q["question"],
                    "is_correct": is_correct,
                    "correct_answer": correct_answer,
                    "user_answer": user_answer,
                    "options": q.get("options", {}),
                }
            )

        st.markdown("---")
        st.success(f"🎯 You scored **{correct}/{total}** ({correct/total*100:.1f}%)")
        st.progress(correct / total if total > 0 else 0)

        st.markdown("### 📊 Quiz Results Review")
        for result in results:
            status = "✅" if result["is_correct"] else "❌"
            user_ans_text = (
                result["options"].get(result["user_answer"], "Not answered")
                if result["user_answer"]
                else "Not answered"
            )
            correct_ans_text = result["options"].get(result["correct_answer"], "N/A")

            st.markdown(f"**Q{result['question_num']}.** {status} {result['question']}")
            st.markdown(f"**Your answer:** {user_ans_text}")
            st.markdown(
                f"**Correct answer:** {result['correct_answer']}) {correct_ans_text}"
            )
            st.markdown("---")

    st.markdown("---")

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "⬅ Back to Study Assistant", key="back_from_quiz", use_container_width=True
        ):
            st.session_state.current_page = "main"
            st.rerun()

    with col2:
        if st.button(
            "🔄 New Quiz",
            key="new_quiz",
            use_container_width=True,
            type="secondary",
        ):
            del st.session_state.quiz_data
            st.rerun()


# -------------------- MAIN APP CLASS --------------------
class PolicyExplainerApp:
    """
    Streamlit-based GenAI assistant for:
    - Policy document explanation (Professional Mode)
    - Study material summarization, Q&A, and Quick Notes (Fresher Mode)
    """

    def __init__(self):
        self.setup_page()
        self.document_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()
        self.initialize_session_state()

    # -------------------- PAGE SETUP --------------------
    def setup_page(self):
        st.set_page_config(
            page_title="AI Professional & Fresher Assistant",
            page_icon="📚",
            layout="wide",
        )
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.title("📚 AI Professional & Fresher Assistant")
        st.caption("Understand, summarize, and interact with documents effortlessly.")
        st.markdown("---")

    # -------------------- SESSION STATE --------------------
    def initialize_session_state(self):
        defaults = {
            "policy_processed": False,
            "policy_summary": "",
            "policy_vector_store": None,
            "policy_chat_history": [],
            "policy_filename": "",
            "study_processed": False,
            "study_summary": "",
            "study_vector_store": None,
            "study_chat_history": [],
            "study_filename": "",
            "flashcard_data": None,
            "quiz_data": None,
            "current_page": "main",
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # -------------------- DOCUMENT PROCESSING --------------------
    def process_uploaded_document(self, uploaded_file, tab_name):
        try:
            with st.spinner(f"Processing your {tab_name.lower()} document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    text = self.document_processor.load_document(tmp_path)
                    chunks = self.document_processor.chunk_document(
                        text, metadata={"filename": uploaded_file.name, "tab": tab_name}
                    )
                    vector_store = self.vector_manager.create_vector_store(chunks)
                    summarizer = PolicySummarizer()
                    summary = summarizer.summarize_document(text)

                    if tab_name.lower() == "professional":
                        st.session_state.policy_processed = True
                        st.session_state.policy_summary = summary
                        st.session_state.policy_vector_store = vector_store
                        st.session_state.policy_filename = uploaded_file.name
                    else:
                        st.session_state.study_processed = True
                        st.session_state.study_summary = summary
                        st.session_state.study_vector_store = vector_store
                        st.session_state.study_filename = uploaded_file.name

                    return True
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False

    # -------------------- UPLOAD SECTION --------------------
    def render_upload_section(self, tab_name):
        st.header(f"📤 Upload Your {tab_name} Document")
        uploaded_file = st.file_uploader(
            f"Choose a {tab_name.lower()} document (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            key=f"uploader_{tab_name.lower()}",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            if st.button(
                f"🚀 Process {tab_name} Document", key=f"process_{tab_name.lower()}"
            ):
                if self.process_uploaded_document(uploaded_file, tab_name):
                    st.success(f"✅ {tab_name} document processed successfully!")
                    st.rerun()

    # -------------------- SUMMARY + QUESTIONS + QUICK NOTES --------------------
    def render_summary_and_questions(self, tab_name):
        processed = (
            st.session_state.policy_processed
            if tab_name.lower() == "professional"
            else st.session_state.study_processed
        )
        if not processed:
            return

        st.markdown("### 📄 Document Summary and Supplementary Q&A")
        col1, col2 = st.columns([2.5, 1.5], gap="large")

        # Left Column (Summary)
        with col1:
            st.markdown("#### 📝 Summary")
            summary = (
                st.session_state.policy_summary
                if tab_name.lower() == "professional"
                else st.session_state.study_summary
            )
            st.markdown(
                f'<div class="main-card">{summary}</div>', unsafe_allow_html=True
            )

        # Right Column (Q&A + Quick Notes)
        with col2:
            st.markdown("#### 🧠 Supplementary Question Box")
            vector_store = (
                st.session_state.policy_vector_store
                if tab_name.lower() == "professional"
                else st.session_state.study_vector_store
            )
            chat_history = (
                st.session_state.policy_chat_history
                if tab_name.lower() == "professional"
                else st.session_state.study_chat_history
            )

            question = st.text_input(
                "Ask a follow-up question:",
                key=f"supplementary_input_{tab_name.lower()}",
            )

            if (
                st.button("Ask", key=f"supplementary_button_{tab_name.lower()}")
                and question
            ):
                with st.spinner("Thinking..."):
                    try:
                        qa_system = PolicyQASystem(vector_store)
                        response = qa_system.ask_question(question)
                        st.markdown(response["answer"])
                        chat_history.append({"role": "user", "content": question})
                        chat_history.append(
                            {"role": "assistant", "content": response["answer"]}
                        )
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

            # Quick Notes Logic (Right Side for Fresher Mode)
            if tab_name.lower() == "fresher":
                st.markdown("---")
                st.markdown("#### 📘 Quick Notes Generator")

                if (
                    "flashcard_data" in st.session_state
                    and st.session_state.flashcard_data
                ):
                    st.success("✅ Quick Notes generated successfully!")

                    # Create two columns for buttons in the main app
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button(
                            "📖 View Quick Notes",
                            key="view_flashcards",
                            use_container_width=True,
                        ):
                            st.session_state.current_page = "quick_notes"
                            st.rerun()

                    with btn_col2:
                        if st.button(
                            "🎯 Take Quiz",
                            key="take_quiz_main",
                            use_container_width=True,
                            type="secondary",
                        ):
                            st.session_state.current_page = "quiz"
                            st.rerun()

                else:
                    if st.button("⚡ Generate Quick Notes", key="generate_flashcards"):
                        with st.spinner("Generating your quick notes..."):
                            try:
                                flashcard_gen = StudyFlashcardGenerator()
                                notes = flashcard_gen.generate_flashcards(
                                    st.session_state.study_summary
                                )
                                st.session_state.flashcard_data = notes
                                st.success("✅ Quick Notes generated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error generating quick notes: {str(e)}")

    # -------------------- PROFESSIONAL TAB --------------------
    def render_professional_tab(self):
        st.header("🏢 Professional Assistant")
        st.markdown(
            "Upload long professional documents to get clear, concise explanations and context-aware Q&A."
        )
        st.markdown("---")

        if not st.session_state.policy_processed:
            self.render_upload_section("Professional")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(
                    f"📄 **Active Document:** {st.session_state.policy_filename}"
                )
            with col2:
                if st.button("🔄 Upload New Document", key="new_policy"):
                    st.session_state.policy_processed = False
                    st.session_state.policy_summary = ""
                    st.session_state.policy_vector_store = None
                    st.session_state.policy_chat_history = []
                    st.session_state.policy_filename = ""
                    st.rerun()
            self.render_summary_and_questions("Professional")

    # -------------------- FRESHER TAB --------------------
    def render_fresher_tab(self):
        st.header("🎓 Fresher Assistant")
        st.markdown(
            "Upload study materials to generate summaries, ask supplementary questions, and view Quick Notes for revision."
        )
        st.markdown("---")

        if not st.session_state.study_processed:
            self.render_upload_section("Fresher")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(
                    f"📄 **Active Study Material:** {st.session_state.study_filename}"
                )
            with col2:
                if st.button("🔄 Upload New Material", key="new_study"):
                    st.session_state.study_processed = False
                    st.session_state.study_summary = ""
                    st.session_state.study_vector_store = None
                    st.session_state.study_chat_history = []
                    st.session_state.study_filename = ""
                    st.session_state.flashcard_data = None
                    st.session_state.quiz_data = None
                    st.rerun()
            self.render_summary_and_questions("Fresher")

    # -------------------- MAIN RUN --------------------
    def run(self):
        # Check current page
        if st.session_state.current_page == "quick_notes":
            quick_notes_page()
            return
        elif st.session_state.current_page == "quiz":
            quiz_page()
            return

        # Main app page
        tab1, tab2 = st.tabs(["🏢 Professional", "🎓 Fresher"])
        with tab1:
            self.render_professional_tab()
        with tab2:
            self.render_fresher_tab()


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    try:
        config.validate_config()
        app = PolicyExplainerApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("💡 Please check your configuration and dependencies.")
