import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

from core.chat_memory import add_chat_memory
from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

def get_llm():

    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        ),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


# --------------------------------------------------
# Format Retrieved Documents
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# --------------------------------------------------
# Build RAG Chain
# --------------------------------------------------

def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting assistant.

Answer the user's question based ONLY on
the meeting transcript context provided below.

If the answer is not found in the context,
say:

"I could not find this information in the meeting transcript."

Always be concise and precise.

If quoting someone, mention it clearly.

Context from meeting transcript:

{context}
""",
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{question}"
            ),
        ]
    )

    rag_chain = (
        {
            "context":
                itemgetter("question")
                | retriever
                | RunnableLambda(format_docs),

            "question":
                itemgetter("question"),

            "chat_history":
                itemgetter("chat_history"),
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    return add_chat_memory(rag_chain)


# --------------------------------------------------
# Load Existing RAG
# --------------------------------------------------

def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting assistant.

Answer the user's question based ONLY on
the meeting transcript context provided below.

If the answer is not found in the context,
say:

"I could not find this information in the meeting transcript."

Always be concise and precise.

Context from meeting transcript:

{context}
""",
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{question}"
            ),
        ]
    )

    rag_chain = (
        {
            "context":
                itemgetter("question")
                | retriever
                | RunnableLambda(format_docs),

            "question":
                itemgetter("question"),

            "chat_history":
                itemgetter("chat_history"),
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    return add_chat_memory(rag_chain)


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(
    rag_chain,
    question: str
) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": "meeting_chat"
            }
        },
    )

    print(f"Answer: {answer}")

    return answer