from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import VectorStore


class KakaoSyncChatBot:
    def __init__(self, db: VectorStore):
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5
        )
        memory = ConversationBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            memory=memory,
            verbose=True
        )

    def ask_question(self, question) -> str:
        return self.conversation(question)['answer']
