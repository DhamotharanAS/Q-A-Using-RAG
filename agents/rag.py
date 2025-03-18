from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os
import getpass
from dotenv import load_dotenv
load_dotenv()

async def loaders():
    loader = PyPDFLoader(
    file_path = "C:\\Users\\2276706\\OneDrive - Cognizant\\Desktop\\Project\\agents\\The_adv 2.pdf",
    mode = "single",
    pages_delimiter = "\n")    

    docs = []
    docs_lazy = loader.lazy_load()
    for doc in docs_lazy:
        docs.append(doc)
    docs = await loader.aload()
    print("Extraction Completed")
    return docs



def chunking(a):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,)
    page_contents = [doc.page_content for doc in a]
    texts = text_splitter.create_documents(page_contents)
    print("Chunking Completed")
    return texts

# chunks = chunking(text)


def embedding(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = faiss.IndexFlatL2(len(embeddings_model.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("Embedding set up completed")
    uuids = [str(uuid4()) for _ in range(len(chunks))]

    vector_store.add_documents(documents=chunks, ids=uuids)
    print("Embedding Stored in VDB")

    results = vector_store.similarity_search(
        "What is the job of sarah",
        k=5
    )
    search_results = [f"* {res.page_content} [{res.metadata}]" for res in results]
    
    print("Search Completed")
    return search_results

# similarity = embedding(chunks)

def ques_ans(similarity):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            api_key=api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    print("Model Loaded")
    Context = similarity
    system_prompt = (
                "You are a Q&A chat bot. Answer the question based only on the provided context "
                "with a minimum of 200 words."
            )
    question = "What is the job of sarah"
    prompt = f"{system_prompt}\n\nContext: {Context}\n\nQuestion: {question}"
    ai_msg = llm.invoke(prompt)
    return ai_msg.content

similarity = '''["* experienced and the unforgettable characters she had met along the way. \n## Chapter 6: The Jewel's Secret \nUpon her return, Sarah discovered another secret about the jewel. It had the power to grant a single \nwish to its holder. With a selfless heart, Sarah wished for the prosperity and happiness of her town. \nAs soon as the wish was made, the jewel vanished, its magic spent but its purpose fulfilled. \n## Chapter 7: A New Beginning [{}]", "* The mysterious letter piqued Sarah's curiosity. The thought of a hidden jewel, an unsolved mystery, \nand an adventure waiting to unfold was too exciting to ignore. She felt a surge of anticipation and \nexcitement - this was the kind of adventure she had always dreamed of. \nShe decided to embark on a quest to find this lost jewel. Packing her bag with essentials - a compass, \na map, a notebook, and a heart full of courage, she set off on her journey. The letter served as her [{}]", "* One day, while exploring the dusty, forgotten corners of her grandmother's attic, she stumbled upon \nan old, yellowed letter. It was hidden in a rusty tin box, under a pile of antiquated trinkets that \nsmelled of time and nostalgia. The envelope bore her grandmother's name, a woman of great \nwisdom and kindness who had passed away many years ago. Sarah remembered her grandmother's \nstories of adventure and mystery, and she felt a strange connection to the past as she held the letter [{}]", "* had found in her grandmother's attic. The dragon, surprised and pleased by her respect and bravery, \nallowed her to take the jewel. \n## Chapter 5: The Return Home \nWith the radiant jewel safely tucked in her bag, Sarah embarked on the journey home. She traversed \nthe same forests and mountains, but this time, they seemed less daunting, more familiar. She \nrealized that the real treasure was not the jewel itself, but the thrilling adventure she had [{}]", '* Life in Serenity flourished like never before. The fields were greener, the harvests were richer, and \nthe people were happier. Sarah, with her brave heart and adventurous spirit, was hailed as a hero. \nAnd though the jewel was gone, its legacy lived on, bringing joy and prosperity to all. \nThe End. [{}]']'''
qa = ques_ans(similarity)
print(qa)



