import sys
import os
import tempfile

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from core.summarizer import analyze_meeting
from core.rag_engine import build_rag_chain, ask_question
from core.transcriber import transcribe_all
from utils.audio_processor import process_input


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎙️ AI Meeting Assistant")

st.write(
    "Upload a meeting recording or provide a YouTube URL "
    "to generate a transcript, summary, action items, "
    "decisions, and open questions."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    language = st.selectbox(
        "Meeting Language",
        ["English", "Hinglish"],
    )

    st.divider()

    st.info(
        "English meetings use local Whisper transcription. "
        "Hinglish meetings use Sarvam AI."
    )


# --------------------------------------------------
# Input Selection
# --------------------------------------------------

input_type = st.radio(
    "Choose input type:",
    ["YouTube URL", "Upload File"],
    horizontal=True,
)


source = None


# --------------------------------------------------
# YouTube URL
# --------------------------------------------------

if input_type == "YouTube URL":

    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if youtube_url:
        source = youtube_url


# --------------------------------------------------
# File Upload
# --------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload audio or video",
        type=[
            "mp3",
            "wav",
            "m4a",
            "mp4",
            "webm",
            "mov",
            "flac",
        ],
    )

    if uploaded_file:

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_file.close()

        source = temp_file.name


# --------------------------------------------------
# Process Meeting
# --------------------------------------------------

process_button = st.button(
    "🚀 Process Meeting",
    type="primary",
    use_container_width=True,
)


if process_button:

    if not source:

        st.warning(
            "Please provide a YouTube URL or upload a file."
        )

    else:

        try:

            # ------------------------------------------
            # Step 1: Audio Processing
            # ------------------------------------------

            with st.status(
                "Processing meeting...",
                expanded=True,
            ) as status:

                st.write(
                    "🎵 Downloading / converting audio..."
                )

                chunks = process_input(source)

                st.write(
                    f"✅ Audio ready — {len(chunks)} "
                    f"chunk(s) created."
                )


                # --------------------------------------
                # Step 2: Transcription
                # --------------------------------------

                st.write(
                    "🎤 Transcribing meeting..."
                )

                transcript = transcribe_all(
                    chunks,
                    language.lower(),
                )

                st.write(
                    "✅ Transcription completed."
                )


                # --------------------------------------
                # Step 3: Meeting Analysis
                # --------------------------------------

                st.write(
                    "🧠 Analyzing meeting with Mistral..."
                )

                # ONE MISTRAL REQUEST
                analysis = analyze_meeting(
                    transcript
                )

                title = analysis["title"]
                summary = analysis["summary"]
                action_items = analysis["action_items"]
                decisions = analysis["key_decisions"]
                questions = analysis["open_questions"]

                st.write(
                    "✅ Meeting analysis completed."
                )


                # --------------------------------------
                # Step 4: Build RAG
                # --------------------------------------

                st.write(
                    "🔎 Building meeting knowledge base..."
                )

                rag_chain = build_rag_chain(
                    transcript
                )

                st.write(
                    "✅ RAG system ready."
                )


                status.update(
                    label="Meeting processed successfully!",
                    state="complete",
                    expanded=False,
                )


            # ------------------------------------------
            # Save Results
            # ------------------------------------------

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
            }

            st.session_state.rag_chain = rag_chain

            # Clear previous chat
            st.session_state.messages = []

            st.success(
                "Meeting processed successfully! 🎉"
            )


        except Exception as e:

            st.error(
                f"Error while processing meeting: {e}"
            )


# --------------------------------------------------
# Display Results
# --------------------------------------------------

if st.session_state.result:

    result = st.session_state.result


    # ----------------------------------------------
    # Meeting Title
    # ----------------------------------------------

    st.header(
        f"📌 {result['title']}"
    )


    # ----------------------------------------------
    # Tabs
    # ----------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Summary",
            "📝 Transcript",
            "✅ Action Items",
            "🔑 Decisions",
            "❓ Questions",
        ]
    )


    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    with tab1:

        st.subheader("Meeting Summary")

        st.markdown(
            result["summary"]
        )


    # ----------------------------------------------
    # Transcript
    # ----------------------------------------------

    with tab2:

        st.subheader("Full Transcript")

        st.text_area(
            "Transcript",
            result["transcript"],
            height=500,
            label_visibility="collapsed",
        )


        st.download_button(
            label="⬇️ Download Transcript",
            data=result["transcript"],
            file_name="meeting_transcript.txt",
            mime="text/plain",
        )


    # ----------------------------------------------
    # Action Items
    # ----------------------------------------------

    with tab3:

        st.subheader("Action Items")

        st.markdown(
            result["action_items"]
        )


    # ----------------------------------------------
    # Decisions
    # ----------------------------------------------

    with tab4:

        st.subheader("Key Decisions")

        st.markdown(
            result["key_decisions"]
        )


    # ----------------------------------------------
    # Questions
    # ----------------------------------------------

    with tab5:

        st.subheader("Open Questions")

        st.markdown(
            result["open_questions"]
        )


    # --------------------------------------------------
    # RAG Chat
    # --------------------------------------------------

    st.divider()

    st.header("💬 Chat With Your Meeting")

    st.write(
        "Ask questions about the meeting transcript."
    )


    # ----------------------------------------------
    # Display Previous Messages
    # ----------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ----------------------------------------------
    # Chat Input
    # ----------------------------------------------

    question = st.chat_input(
        "Ask something about the meeting..."
    )


    if question:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the meeting..."
            ):

                try:

                    answer = ask_question(
                        st.session_state.rag_chain,
                        question,
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Error while answering: {e}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )