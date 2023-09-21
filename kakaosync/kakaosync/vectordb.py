from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = 'vectordb/chroma-persist'
CHROMA_COLLECTION_NAME = 'dosu-bot'

vectordb = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(),
    persist_directory=CHROMA_PERSIST_DIR
)
