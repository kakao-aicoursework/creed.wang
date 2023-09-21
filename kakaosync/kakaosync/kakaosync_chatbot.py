import asyncio

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import VectorStore

CUSTOM_QA_PROMPT_TEMPLATE = """아래 Context를 활용해서 Question에 대한 답변을 해줘.
카카오소셜, 카카오싱크, 카카오톡채널에 관련된 대답만 하고, 그 외에는 답변할 수 없다고 대답해.

<context>
{context}
</context>

<Question>
{question}
</Question>
Answer:
"""
CUSTOM_QA_PROMPT = PromptTemplate(
    template=CUSTOM_QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


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
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    async def ask_question(self, question):
        callback_handler = AsyncIteratorCallbackHandler()
        question_llm = ChatOpenAI(temperature=0, verbose=True)
        question_generator = LLMChain(
            llm=question_llm,
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=True
        )
        streaming_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            streaming=True,
            callbacks=[callback_handler],
            temperature=0
        )
        doc_chain = load_qa_chain(
            streaming_llm,
            chain_type="stuff",
            prompt=CUSTOM_QA_PROMPT,
            verbose=True
        )
        self.conversation = ConversationalRetrievalChain(
            combine_docs_chain=doc_chain,
            retriever=self.db.as_retriever(),
            question_generator=question_generator,
            memory=self.memory,
            verbose=True
        )
        task = asyncio.create_task(
            self._aask_question(question, callback_handler.done),
        )
        async for token in callback_handler.aiter():
            yield token
        await task

    async def _aask_question(self, question, event):
        try:
            await self.conversation.ainvoke(question)
        finally:
            event.set()
