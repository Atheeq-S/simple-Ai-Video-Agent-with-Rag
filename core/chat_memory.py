from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def add_chat_memory(chain):
    histories = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()
        return histories[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )