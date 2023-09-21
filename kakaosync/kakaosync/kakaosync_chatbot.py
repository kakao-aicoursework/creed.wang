import asyncio

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import VectorStore


class KakaoSyncChatBot:
    """
    Written with reference of below documents
    https://gist.github.com/ninely/88485b2e265d852d3feb8bd115065b1a
    https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db

    Look up this prompts
    langchain.chains.conversational_retrieval.prompts.CONDENSE_QUESTION_PROMPT
    langchain.chains.conversational_retrieval.prompts.QA_PROMPT
    """
    def __init__(self, db: VectorStore):
        self.db = db
        self.question_llm = ChatOpenAI(temperature=0, verbose=True)
        self.memory = ConversationBufferMemory(
            llm=self.question_llm,
            memory_key="chat_history",
            return_messages=True
        )

    async def ask_question(self, question):
        self.callback = AsyncIteratorCallbackHandler()
        streaming_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            streaming=True,
            callbacks=[self.callback],
            temperature=0
        )
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=streaming_llm,
            retriever=self.db.as_retriever(),
            condense_question_llm=self.question_llm,
            memory=self.memory,
            verbose=True
        )
        task = asyncio.create_task(
            self._aask_question(question, self.callback.done),
        )
        async for token in self.callback.aiter():
            yield token
        await task

    async def _aask_question(self, question, event):
        try:
            await self.conversation.ainvoke(question)
        finally:
            event.set()
