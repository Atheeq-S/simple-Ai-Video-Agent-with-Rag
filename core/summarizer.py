import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        temperature=0.2,
    )


# --------------------------------------------------
# Analyze Meeting
# --------------------------------------------------

def analyze_meeting(transcript: str) -> dict:

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting analyst.

Analyze the following meeting transcript.

Return the result in EXACTLY this format:

TITLE:
<short professional meeting title, maximum 8 words>

SUMMARY:
- <important point>
- <important point>
- <important point>

ACTION ITEMS:
1. Task: <task>
   Owner: <person if mentioned, otherwise Not specified>
   Deadline: <deadline if mentioned, otherwise Not specified>

KEY DECISIONS:
1. <decision>

OPEN QUESTIONS:
1. <unresolved question or follow-up topic>

Rules:

- Use ONLY information from the transcript.
- Do not invent information.
- Do not invent names or deadlines.
- Keep the summary concise.
- If there are no action items, write:
  No action items found.
- If there are no key decisions, write:
  No key decisions found.
- If there are no open questions, write:
  No open questions found.
""",
            ),
            (
                "human",
                "{transcript}",
            ),
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    print("Sending transcript to Groq...")

    result = chain.invoke(
        {
            "transcript": transcript
        }
    )

    print("Groq analysis completed.")

    return parse_analysis(result)


# --------------------------------------------------
# Parse Groq Response
# --------------------------------------------------

def parse_analysis(result: str) -> dict:

    sections = {
        "title": "",
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "open_questions": "",
    }

    current_section = None

    for line in result.splitlines():

        line = line.strip()

        if not line:
            continue

        upper_line = line.upper()

        if upper_line == "TITLE:":
            current_section = "title"

        elif upper_line == "SUMMARY:":
            current_section = "summary"

        elif upper_line == "ACTION ITEMS:":
            current_section = "action_items"

        elif upper_line == "KEY DECISIONS:":
            current_section = "key_decisions"

        elif upper_line == "OPEN QUESTIONS:":
            current_section = "open_questions"

        elif current_section:

            sections[current_section] += line + "\n"

    return {
        key: value.strip()
        for key, value in sections.items()
    }