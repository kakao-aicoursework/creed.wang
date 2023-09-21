import os

import openai
import reflex as rx

from kakaosync.kakaosync_chatbot import KakaoSyncChatBot
from kakaosync.vectordb import vectordb

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
kakaosync_chatbot = KakaoSyncChatBot(vectordb)


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = {
        "Intros": [],
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
        self.chats[self.new_chat_name] = []
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
                "New Chat": []
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
        # Check if we have already asked the last question
        # or if the question is empty
        self.question = form_data["question"]
        if self._isEmptyOrSameQuestion():
            return

        # Set the processing flag to true and yield.
        self.processing = True
        yield

        qa = QA(question=self.question, answer="")
        self.chats[self.current_chat].append(qa)
        self.chats = self.chats
        async for answer_text in kakaosync_chatbot.ask_question(self.question):
            self.chats[self.current_chat][-1].answer += answer_text
            self.chats = self.chats
            yield

        # Toggle the processing flag.
        self.processing = False

    def _isEmptyOrSameQuestion(self) -> bool:
        if len(self.chats[self.current_chat]) == 0:
            return
        return (self.chats[self.current_chat][-1].question == self.question or
                self.question == "")
