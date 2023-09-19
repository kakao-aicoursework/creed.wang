import os

import openai
import reflex as rx
from kakaosync.kakaosync_chatbot import KakaoSyncChatBot

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
kakaosync_chatbot = KakaoSyncChatBot()


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = {
        "Intros": [QA(question="What is your name?", answer="reflex")],
    }

    # The current chat name.
    current_chat = "Intros"

    # The currrent question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    # Whether the drawer is open.
    drawer_open: bool = False

    # Whether the modal is open.
    modal_open: bool = False

    def create_chat(self):
        """Create a new chat."""
        # Insert a default question.
        self.chats[self.new_chat_name] = [
            QA(question="What is your name?", answer="reflex")
        ]
        self.current_chat = self.new_chat_name

    def toggle_modal(self):
        """Toggle the new chat modal."""
        self.modal_open = not self.modal_open

    def toggle_drawer(self):
        """Toggle the drawer."""
        self.drawer_open = not self.drawer_open

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = {
                "New Chat": [QA(question="What is your name?", answer="reflex")]
            }
        self.current_chat = list(self.chats.keys())[0]
        self.toggle_drawer()

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name
        self.toggle_drawer()

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """
        # Check if we have already asked the last question or if the question is empty
        self.question = form_data["question"]
        if (
            self.chats[self.current_chat][-1].question == self.question
            or self.question == ""
        ):
            return

        # Set the processing flag to true and yield.
        self.processing = True
        yield

        # Build the messages.
        messages = [{
            "role": "system",
            "content": ("너는 아래 카카오싱크에 대한 소개 문서를 기반으로 "
                        "카카오싱크에 대한 질문에 대답하는 assistant야.\n"
                        "카카오싱크와 관련없는 질문에는 대답하지 마\n"
                        "---\n")
        }]

        for qa in self.chats[self.current_chat][1:]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        messages.append({"role": "user", "content": self.question})

        # Start a new session to answer the question.
        # session = openai.ChatCompletion.create(
        #     model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        #     messages=messages,
        #     # max_tokens=50,
        #     # n=1,
        #     stop=None,
        #     temperature=0.7,
        #     stream=True,  # Enable streaming
        # )

        answer = kakaosync_chatbot.ask_question(self.question)
        qa = QA(question=self.question, answer=answer)
        self.chats[self.current_chat].append(qa)

        # Stream the results, yielding after every word.
        # for item in session:
        #     if hasattr(item.choices[0].delta, "content"):
        #         answer_text = item.choices[0].delta.content
        #         self.chats[self.current_chat][-1].answer += answer_text
        #         self.chats = self.chats
        #         yield

        # Toggle the processing flag.
        self.processing = False