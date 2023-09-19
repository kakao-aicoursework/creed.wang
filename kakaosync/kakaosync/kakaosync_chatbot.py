from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class KakaoSyncChatBot:
    def __init__(self):
        loader = TextLoader(file_path="assets/kakaosync.txt")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0
        )
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings()
        )
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5
        )
        retriever = vectorstore.as_retriever()
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

    def ask_question(self, question) -> str:
        return self.conversation(question)['answer']
