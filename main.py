import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from typing import List

# OpenAIのAPIキーの設定
os.environ["OPENAI_API_KEY"] = 'OPEN_AI_API_KEYを入力'

def create_qa_system(texts: List[str]):
    # テキストの分割
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_texts = []
    for text in texts:
        split_texts.extend(text_splitter.split_text(text))

    # テキストデータのベクトル化とFAISSインデックスの作成
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(split_texts, embeddings)

    # LangchainのQ&Aチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity"),
    )

    return qa_chain

def answer_question(qa_chain, question: str):
    response = qa_chain({"query": question})
    return response['result']

# 使用例
if __name__ == "__main__":
    # テキストのリストを作成
    texts = [
        """パンや小麦粉をよく食べると太りやすくなります。""",
        
        """スーパーの夜半額セールは衝動買いをしやすく、予定以上に食べてしまいます。""",
        
        """先ずはこまめな運動から、難しければ軽いジョギングなどから始めるのがダイエットには効果的です。"""
    ]

    # QAシステムの作成
    qa_system = create_qa_system(texts)

    # 質問と回答
    question = "ダイエットが続かないのですがどうしたらよいでしょうか？"
    answer = answer_question(qa_system, question)
    print(answer)