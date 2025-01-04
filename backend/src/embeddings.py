#Importing the python dependencies
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader

class Embeddings:
    """
    A class to handle text embedding operations using a specified LLM endpoint and ChromaDB for storage.

    Attributes:
        url (str): The base URL for the LLM endpoint.
        embeddings (OllamaEmbeddings): An embedding model initialized with the specified LLM URL.
        file_path (str): Path to the file containing the document to be embedded.
        persist_directory (str): Directory where ChromaDB embeddings are stored.

    Methods:
        get_chunks(chunk_size: int, chunk_overlap: int, company_name: str) -> list:
            Splits a document into chunks and associates metadata with each chunk.

        load_documents(chunk_size: int, chunk_overlap: int, company_name: str, use_existing: bool) -> Chroma:
            Loads document chunks into ChromaDB, either creating a new database or appending to an existing one.

        get_vector_store() -> Chroma:
            Loads and returns the ChromaDB vector store from the persist directory.

        get_retriever(vector_store: Chroma, k: int) -> Retriever:
            Returns a retriever for performing similarity searches on the vector store.
    """
    def __init__(self, llm_url, persist_directory="data/") -> None:
        """
        Initializes the Embeddings class.

        Args:
            llm_url (str): The base URL for the LLM endpoint.
            persist_directory (str, optional): Directory for storing ChromaDB embeddings. Defaults to "data/".
        """
        self.url = llm_url
        self.embeddings = OllamaEmbeddings(base_url=self.url, model="nomic-embed-text")
        self.persist_directory = persist_directory

    def get_chunks(self, file_path, chunk_size, chunk_overlap, company_name):
        """
        Splits a document into smaller chunks and adds metadata.

        Args:
            file_path (str): Path to the file to be processed.
            chunk_size (int): Size of each chunk in characters.
            chunk_overlap (int): Overlap size between chunks in characters.
            company_name (str): Metadata to associate with each chunk.

        Returns:
            list: A list of document chunks with metadata.
        """
        document = PyPDFLoader(file_path=file_path).load()
        document_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap,
                                                        length_function=len,
                                                        )
        chunks = document_splitter.split_documents(document)
        for i in range(len(chunks)):
            chunks[i].metadata["company_name"] = company_name
        return chunks

    def load_documents(self, file_path, chunk_size, chunk_overlap, company_name, 
                       use_existing):
        """
        Loads document chunks into ChromaDB.

        Args:
            file_path (str): Path to the file to be processed.
            chunk_size (int): Size of each chunk in characters.
            chunk_overlap (int): Overlap size between chunks in characters.
            company_name (str): Metadata to associate with each chunk.
            use_existing (bool): If True, appends to an existing ChromaDB. Otherwise, creates a new one.

        Returns:
            Chroma: The vector store containing the document embeddings.
        """
        chunks = self.get_chunks(file_path=file_path, chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap, 
                                 company_name=company_name)
        if not use_existing:
            vector_store = Chroma.from_documents(documents=chunks, 
                                                embedding=self.embeddings, 
                                                persist_directory=self.persist_directory)
        else:
            vector_store = Chroma(persist_directory=self.persist_directory, 
                                  embedding_function=self.embeddings)
            vector_store.add_documents(chunks)
            vector_store.persist()
        return vector_store

    def get_vector_store(self):
        """
        Loads the ChromaDB vector store from the persist directory.

        Returns:
            Chroma: The vector store.
        """
        vector_store = Chroma(persist_directory=self.persist_directory, 
                                embedding_function=self.embeddings)
        return vector_store

    def get_retriever(self, vector_store, k):
        """
        Creates a retriever for performing similarity searches on the vector store.

        Args:
            vector_store (Chroma): The vector store to perform searches on.
            k (int): Number of top results to return for each search query.

        Returns:
            Retriever: A retriever object for similarity search.
        """
        retriever = vector_store.as_retriever(search_type="similarity",
            search_kwargs={
                "k": k,
                #"score_threshold": 0.1,
            },
        )
        return retriever