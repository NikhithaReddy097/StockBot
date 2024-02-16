from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

DATA_PATH = "Data"
DB_FAISS_PATH = "vector_store"
MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"


def get_embeddings():
    # loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls = PyPDFLoader)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 100)
    # texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_FAISS_PATH)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    print("Db created")
    return db

def get_LLM():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=15,
        n_batch=256,
        n_ctx = 4096,
        verbose=True,  
        callbackManager= [StdOutCallbackHandler()],
    )
    print("llm_created")
    return llm

def get_chain(llm, prompt, db):
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={'k':10}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt}
    )
    # chain = ConversationalRetrievalChain.from_llm(
    #         llm = llm,
    #     chain_type = "stuff",
    #     retriever = db.as_retriever(search_kwargs={'k':10}),
    #     return_source_documents = True,
    #     chain_type_kwargs = {'prompt': prompt}
    # )
    print("chain created")
    return chain

def get_prompt():
    template = """You are a chatbot having a conversation with a human. 
1. Given the following extracted parts of a long document and a question, create a final answer. 
2. Review the context very carefully to understand the details and nuances of the information provided and then use that content to generate the answer. 
3. Analyze the question to determine what specific information is being requested. 
4. Search the context for the information that directly answers the question. 
5. If the answer is found within the context, provide the answer clearly and concisely. 
6. If the answer is not found within the context, reply with "I do not know". 
7. Never make up any questions by yourselves, just answer the question given by user. 
8. Do not mention "Chatbot response:" in the result. 
9. When formulating a question, don't reference the provided document or say "from the provided context", "as described in the document", "according to the given document" or anything similar. 
Context:{context} 
Question:{question} 
Please return only correct answer. If you do not find the relevant documents just reply with "I do not know". 
Helpful answer:"""
    prompt = PromptTemplate(
                template=template, 
                input_variables = ['context', 'question']
            )
    return prompt