'''
    RAG主函数
'''
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from CustomLLM import CustomLLM


class Rag:

    def __init__(self, question):
        self.q = question

    def __str__(self):
        if self.q:
            embedding_function = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
            db = Chroma(persist_directory="/", embedding_function=embedding_function) #改成实际的本地向量库地址
            doc = db.similarity_search(self.q)
            context = doc[0].page_content

            #提示词
            prompt_template = """请用以下片段去回答问题. 如果不知道答案, 就说不知道. 不要编造答案.
            {context}
            Question: {question}
            Answer:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            llm = CustomLLM()
            query_llm = LLMChain(llm=llm, prompt=prompt)
            response = query_llm.run({"context": context, "question": self.q})
            return response