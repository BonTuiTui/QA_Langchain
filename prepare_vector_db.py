from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

#khai bao bien
data_path = "data"
vector_db_path = "vector_store/db_faiss"
#func1: Tao ra vector db tu mot doan text

def create_db_from_text():
    raw_text = '''
    The Dursleys had everything they wanted, but they also had a
secret, and their greatest fear was that somebody would discover
it. They didn’t think they could bear it if anyone found out about
the Potters. Mrs Potter was Mrs Dursley’s sister, but they hadn’t
met for several years; in fact, Mrs Dursley pretended she didn’t
have a sister, because her sister and her good-for-nothing husband
were as unDursleyish as it was possible to be. The Dursleys
shuddered to think what the neighbours would say if the Potters
arrived in the street. The Dursleys knew that the Potters had a
small son, too, but they had never even seen him. This boy was
another good reason for keeping the Potters away; they didn’t
want Dudley mixing with a child like that.
    '''
    #chia nho van ban
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    #embedding
    embeddings = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    # Dua vao faisse Vector DB
    db = FAISS.from_texts(texts = chunks,embedding = embeddings)
    db.save_local(vector_db_path)
    return db

create_db_from_text()
























