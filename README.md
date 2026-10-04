
# 🎙️ AI Video Meeting Assistant with RAG

An AI-powered video and meeting assistant that converts meeting recordings or YouTube videos into searchable transcripts and automatically generates **summaries, action items, key decisions, and open questions**.

The system also provides a **Retrieval-Augmented Generation (RAG) chat interface**, allowing users to ask questions about the meeting using natural language.

---

## 🚀 Features

- 🎥 Process YouTube videos
- 📁 Upload local audio/video files
- 🎤 Local speech-to-text using OpenAI Whisper
- 🇮🇳 Hinglish transcription and translation using Sarvam AI
- 🧠 Meeting analysis using Groq
- 📝 Automatic meeting summary
- ✅ Automatic action-item extraction
- 🔑 Key-decision extraction
- ❓ Open-question extraction
- 🔎 Semantic search using ChromaDB
- 💬 Chat with the meeting transcript
- 🧠 Conversation memory for the RAG chatbot
- 🌐 Streamlit web interface
- 📥 Download generated transcripts

---

## 🏗️ System Architecture

```text
                    USER INPUT
                        │
             ┌──────────┴──────────┐
             │                     │
        YouTube URL            Local File
             │                     │
             └──────────┬──────────┘
                        ↓
                Audio Processing
                  yt-dlp / Pydub
                        ↓
                  WAV Conversion
                 Mono / 16 kHz
                        ↓
                  Audio Chunking
                        ↓
              ┌─────────┴─────────┐
              │                   │
           English              Hinglish
              │                   │
          Whisper             Sarvam AI
              │                   │
              └─────────┬─────────┘
                        ↓
                  Full Transcript
                        ↓
                Groq LLM Analysis
                        ↓
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
     Summary       Action Items      Decisions
                        │
                        ↓
                  Open Questions
                        ↓
                    ChromaDB
                        ↓
                  Vector Search
                        ↓
                       RAG
                        ↓
                  Groq LLM Chat
                        ↓
                   User Answer
```

---

## 🛠️ Technologies Used

### Speech-to-Text

**OpenAI Whisper**

Used for local English speech transcription.

```text
Audio → Whisper → English Transcript
```

### Hinglish Processing

**Sarvam AI**

Used for speech-to-text translation for Hinglish meetings.

```text
Hinglish Audio → Sarvam AI → English Transcript
```

### Large Language Model

**Groq**

The project uses Groq for meeting analysis and RAG-based question answering.

Current model:

```text
openai/gpt-oss-20b
```

Groq is accessed through LangChain's `ChatGroq` integration.

### RAG

The Retrieval-Augmented Generation pipeline uses:

- ChromaDB
- HuggingFace sentence embeddings
- LangChain
- Recursive text splitting

### User Interface

**Streamlit**

Provides the web interface for:

- Uploading files
- Entering YouTube URLs
- Viewing transcripts
- Viewing meeting analysis
- Asking questions

---

## 📂 Project Structure

```text
simple-Ai-Video-Agent-with-Rag/
│
├── core/
│   ├── chat_memory.py
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── ui/
│   └── app.py
│
├── downloads/
│
├── vector_db/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> `downloads/` and `vector_db/` are generated during execution and should not be committed to Git.

---

## ⚙️ Requirements

### Software

- Python 3.12
- FFmpeg
- Deno
- Git
- `uv`

Python 3.12 is recommended for compatibility with the project's audio-processing dependencies.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Atheeq-S/simple-Ai-Video-Agent-with-Rag.git
```

Move into the project:

```bash
cd simple-Ai-Video-Agent-with-Rag
```

---

### 2. Create a virtual environment

Using `uv`:

```bash
uv venv --python 3.12
```

Activate it:

```bash
source .venv/bin/activate
```

Verify:

```bash
python --version
```

---

### 3. Install FFmpeg

On macOS:

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

---

### 4. Install Deno

Deno is used by `yt-dlp` for YouTube JavaScript runtime support.

```bash
brew install deno
```

Verify:

```bash
deno --version
```

---

### 5. Install Python dependencies

```bash
uv pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5

WHISPER_MODEL=small
```

### Important

Never commit `.env` to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
downloads/
vector_db/
*.wav
*.mp3
*.mp4
*.m4a
*.webm
.DS_Store
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run ui/app.py
```

The application will open in your browser.

---

## 🎥 Processing a YouTube Video

1. Select **YouTube URL**
2. Enter the YouTube video URL
3. Select the meeting language
4. Click **Process Meeting**

The application performs:

```text
YouTube URL
     ↓
Download Audio
     ↓
Convert to WAV
     ↓
Split into Chunks
     ↓
Whisper / Sarvam
     ↓
Transcript
     ↓
Groq Analysis
     ↓
ChromaDB
     ↓
RAG Chat
```

---

## 📁 Processing a Local File

The application supports audio/video files such as:

```text
MP3
WAV
M4A
MP4
WEBM
MOV
FLAC
```

Select:

```text
Upload File
```

Upload the meeting recording and click:

```text
Process Meeting
```

---

## 🧠 Meeting Analysis

After transcription, Groq analyzes the complete meeting transcript.

The system generates:

### 📋 Summary

Important points discussed during the meeting.

### ✅ Action Items

```text
Task
Owner
Deadline
```

### 🔑 Key Decisions

Important decisions made during the meeting.

### ❓ Open Questions

Unresolved questions and topics requiring follow-up.

---

## 🔎 RAG-Based Meeting Chat

The transcript is divided into smaller chunks and converted into embeddings.

```text
Transcript
     ↓
Text Splitting
     ↓
Embeddings
     ↓
ChromaDB
     ↓
Retriever
     ↓
Relevant Context
     ↓
Groq
     ↓
Answer
```

Users can ask questions such as:

```text
Who is S. Jaishankar?

What was discussed about India?

What were the main decisions?

Who took the interview?

What action items were mentioned?

What topics require follow-up?
```

The chatbot retrieves relevant transcript sections before generating an answer.

---

## 💬 Conversation Memory

The chatbot maintains conversation history using LangChain's in-memory chat history.

This allows follow-up questions such as:

```text
User:
What was discussed about India?

Assistant:
...

User:
Who mentioned it?

Assistant:
...
```

---

## 🧩 Main Components

### `core/transcriber.py`

Responsible for:

- Loading Whisper
- English transcription
- Sarvam AI transcription
- Hinglish processing
- Combining transcript chunks

---

### `core/summarizer.py`

Responsible for:

- Calling Groq
- Generating the meeting title
- Generating the summary
- Extracting action items
- Extracting decisions
- Extracting open questions

---

### `core/vector_store.py`

Responsible for:

- Splitting transcript text
- Creating embeddings
- Creating ChromaDB vector store
- Loading the vector database
- Creating the retriever

---

### `core/rag_engine.py`

Responsible for:

- Retrieving relevant transcript chunks
- Building the RAG prompt
- Sending context to Groq
- Generating answers
- Connecting conversation memory

---

### `core/chat_memory.py`

Responsible for maintaining the chatbot conversation history.

---

### `utils/audio_processor.py`

Responsible for:

- YouTube audio downloading
- Local file conversion
- WAV conversion
- Audio normalization
- Audio chunking

---

### `ui/app.py`

Provides the Streamlit user interface.

---

## 🔄 Processing Pipeline

```text
1. User provides YouTube URL or local file
                    ↓
2. Audio is downloaded / converted
                    ↓
3. Audio converted to mono 16 kHz WAV
                    ↓
4. Audio divided into chunks
                    ↓
5. Whisper / Sarvam transcribes audio
                    ↓
6. Full transcript generated
                    ↓
7. Groq analyzes the transcript
                    ↓
8. Summary and meeting information generated
                    ↓
9. Transcript stored in ChromaDB
                    ↓
10. User asks questions
                    ↓
11. Relevant chunks retrieved
                    ↓
12. Groq generates contextual answer
```

---

## ⚡ Why RAG?

Instead of sending the entire meeting transcript to the LLM for every question, RAG retrieves only the most relevant transcript sections.

This provides:

- More relevant answers
- Lower unnecessary token usage
- Better handling of long meetings
- Grounded answers based on the transcript
- Searchable meeting knowledge

---

## ⚠️ Limitations

- Local Whisper transcription can be slow when running on CPU.
- Long meetings require more processing time.
- YouTube downloading may fail when YouTube changes its access requirements.
- Groq API usage is subject to API limits.
- Sarvam AI requires an API key for Hinglish processing.
- RAG answers depend on the quality of the transcript and retrieved chunks.
- The current chat memory is stored in memory and is not persistent across application restarts.

---

## 🔮 Future Improvements

- Speaker diarization
- Automatic speaker identification
- Timestamp-based transcript navigation
- Persistent chat history
- Multi-meeting knowledge base
- Meeting comparison
- Advanced semantic search
- Automatic email generation for action items
- Calendar integration
- Better multilingual support
- GPU acceleration for Whisper
- Authentication and user accounts

---

## 👨‍💻 Author

**Atheeq S**

GitHub:

https://github.com/Atheeq-S

Repository:

https://github.com/Atheeq-S/simple-Ai-Video-Agent-with-Rag

---

## 📄 License

This project is intended for educational and research purposes.
