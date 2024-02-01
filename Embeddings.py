'''
    嵌入知识库
'''
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import Chroma

filepath = '/***'  #你本地文件及地址
loader = TextLoader(filepath, encoding='utf-8')
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["},\n"],
    length_function=len
)
split_docs = text_splitter.split_documents(text)

embedding_function = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")

vectorstore = Chroma.from_documents(split_docs, embedding_function, persist_directory="/") # 本地存储到向量数据库的地址，换成你自己的
vectorstore.persist()

# 测试是否嵌入成功
query = "你是谁？" #测试问句
doc = vectorstore.similarity_search(query)
print(doc[0].page_content)
