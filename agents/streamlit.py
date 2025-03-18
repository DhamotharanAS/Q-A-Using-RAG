from fastapi import FastAPI, File, UploadFile,HTTPException, Form
from fastapi.responses import FileResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
import asyncio
load_dotenv()
from logger import initialize_logger


class rag:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.llm_name = "gemini-1.5-pro"
        self.vector_store = None
        self.logger = initialize_logger()    
        
    def loaders(self, file_path):
        try:
            loader = PyPDFLoader(
                file_path=file_path,
                mode="single",
                pages_delimiter="\n"
            )

            docs = []
            docs_lazy = loader.lazy_load()
            for doc in docs_lazy:
                docs.append(doc)
            docs = loader.load()
            self.logger.info("Extraction Completed")
            return docs
        except Exception as e:
            return str(e)

    def chunking(self, docs):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            page_contents = [doc.page_content for doc in docs]
            texts = text_splitter.create_documents(page_contents)
            self.logger.info("chunking Completed")
            return texts
        except Exception as e:
            return str(e)

    def embedding(self,chunks):
        try:
            embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            index = faiss.IndexFlatL2(len(embeddings_model.embed_query("hello world")))
            self.vector_store = FAISS(
                embedding_function=embeddings_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            self.vector_store.add_documents(documents=chunks, ids=uuids)
            self.logger.info("Embedding Stored in VDB")
        except Exception as e:
            return str(e)
    def similarity(self,question):
        try:
            results = self.vector_store.similarity_search(
                question,
                k=5
            )
            search_results = [f"* {res.page_content} [{res.metadata}]" for res in results]
            print("Similarity search results:", search_results)
            self.logger.info("similarity Completed")
            return search_results
        except Exception as e:
            return str(e)

    def ques_ans(self, similarity, question):
        try:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            llm = ChatGoogleGenerativeAI(
                model=self.llm_name,
                api_key=self.api_key,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            Context = similarity
            system_prompt = (
                "You are a Q&A chat bot. Answer the question based only on the provided context "
                "with a minimum of 200 words."
            )
            prompt = f"{system_prompt}\n\nContext: {Context}\n\nQuestion: {question}"
            ai_msg = llm.invoke(prompt)
            self.logger.info("Q&A Completed")
            return ai_msg.content
        except Exception as e:
            return str(e)
        
app = FastAPI()

rags = rag()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),question: str = Form(...)):
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    docs = rags.loaders(file_path)
    chunks = rags.chunking(docs)
    rags.embedding(chunks)
    similarity = rags.similarity(question)
    qa = rags.ques_ans(similarity, question)
    

    return {"The Question is": question, "The Answer is": qa}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




