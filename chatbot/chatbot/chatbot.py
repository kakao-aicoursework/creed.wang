"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import os
from datetime import datetime

import openai
import pynecone as pc
from pynecone.base import Base

openai.api_key = os.environ["OPENAI_API_KEY"]


class ChatGptMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_json(self):
        return {
            "role": self.role,
            "content": self.content
        }


class MessageHistory(Base):
    sent_message: str
    received_message: str
    created_at: str


def chat_to_chatgpt(message: str, histories: list[MessageHistory]) -> str:
    chat_gpt_messages = []
    # Add previous messages
    for history in histories:
        chat_gpt_messages.append(
            ChatGptMessage(
                role="user",
                content=history.sent_message
            ).to_json()
        )
        chat_gpt_messages.append(
            ChatGptMessage(
                role="assistant",
                content=history.received_message
            ).to_json()
        )

    # Add new message
    chat_gpt_messages.append(
        ChatGptMessage(
            role="user",
            content=message
        ).to_json()
    )

    # Call ChatGPT Api
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_gpt_messages
    )
    return response['choices'][0]['message']['content']


class State(pc.State):
    """The app state."""

    message: str = ""
    histories: list[MessageHistory] = []
    output: str = "Wait for message..."

    def post(self):
        if not self.message:
            self.output = "Empty message!"
            return

        self.output = chat_to_chatgpt(
            message=self.message,
            histories=self.histories
        )
        self.histories = [
            MessageHistory(
                sent_message=self.message,
                received_message=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ] + self.histories


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("ChatBot 🤖", font_size="2rem"),
        pc.text(
            "Ask anything to chatbot!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def history(history: MessageHistory):
    return pc.box(
        pc.vstack(
            text_box(history.sent_message),
            down_arrow(),
            text_box(history.received_message),
            pc.box(
                pc.text(history.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Ask anything",
            on_blur=State.set_message,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.histories, history),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Chatbot")
app.compile()
