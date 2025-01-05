#Importing the python dependencies
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.chains import ConversationChain
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationBufferMemory
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class AdvancedRAG:
    """
    A class implementing an advanced Retrieval-Augmented Generation (RAG) workflow with document retrieval, grading, 
    response generation, and validation functionalities.

    Attributes:
        url (str): URL of the LLM service.
        llm (ChatGroq): The LLM instance for processing prompts.
        vector_store: The vector store instance for similarity-based document retrieval.
        workflow (StateGraph): A workflow object to manage state transitions in RAG operations.

    Prompts:
        entitlement_prompt: Validates a user's entitlement to access information for specific companies.
        retriever_prompt: Grades the relevance of retrieved documents to a user query.
        rag_generate_prompt: Generates a concise answer based on retrieved context.
        hallucination_prompt: Validates whether the generated answer is grounded in the retrieved context.
        answer_grader_prompt: Assesses the alignment of the generated answer with the user question.

    Methods:
        __init__(llm_url, vector_store):
            Initializes the RAG system with the LLM URL and vector store.

        retrieve(state: dict) -> dict:
            Retrieves documents relevant to a given question and filters them by company name.

        generate(state: dict) -> dict:
            Generates a concise response based on the retrieved documents and the query.

        grade_documents(state: dict) -> dict:
            Grades the relevance of retrieved documents and filters out irrelevant ones.

        default_reply(state: dict) -> dict:
            Provides a default reply when the system cannot generate a valid response.

        grade_generation(state: dict) -> str:
            Grades the generated response for hallucination and relevance to the query.

        add_nodes():
            Adds nodes to the RAG workflow for different stages of processing.

        build_graph() -> StateGraph:
            Builds and compiles the workflow graph for managing RAG operations.

        execute_graph(question: str, company_name: list, email: str) -> dict:
            Executes the RAG workflow for a given question, company names, and user email.

    Example Usage:
        >>> vector_store = YourVectorStoreInstance()
        >>> rag_system = AdvancedRAG(llm_url="http://your-llm-service-url", vector_store=vector_store)
        >>> result = rag_system.execute_graph(
        ...     question="What is the market growth rate for XYZ Corp?",
        ...     company_name=["XYZ Corp"],
        ...     email="user@example.com"
        ... )
    """
    def __init__(self, llm_url, vector_store) -> None:
        """
        Initializes the AdvancedRAG system.

        Args:
            llm_url (str): The URL of the LLM service.
            vector_store: A vector store instance for similarity-based document retrieval.
        """
        self.url = llm_url
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", 
                            temperature=0.0, 
                            max_retries=2)
        self.vector_store = vector_store
        self.workflow = None

        # Prompt to do entitlement check
        entitlement_prompt = PromptTemplate(
            template="""You are a system that extracts a company name from a user question and checks it against a given list of company names.
            Task
            1. Extract the company name mentioned in the question.
            2. Compare the extracted name with the provided list of company names.
            3. Return the result in JSON format:
                "match": "Yes" or "No"
            Here are the company names: \n\n {company_name} \n\n
            Here is the user question: {question} \n >
            """,
            input_variables=["question", "company_name"],
        )

        self.entitlement_chain = entitlement_prompt | self.llm | JsonOutputParser()

        # Prompt to grade the retrieved documents
        retriever_prompt = PromptTemplate(
            template="""You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n >
            """,
            input_variables=["question", "document"],
        )

        self.retrieval_grader = retriever_prompt | self.llm | JsonOutputParser()

        # Prompt to generate final response
        rag_generate_prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise. Do not mention anything about the figures, charts or any visualisation.
            If you do not know the company name then ask the follow-up question to the user to provide the company name.
            Question: {question} 
            Context: {context} 
            Answer: """,
            input_variables=["question", "document"],
        )

        self.rag_chain = rag_generate_prompt | self.llm | StrOutputParser()

        # Prompt to check final response is hallucinated or not
        hallucination_prompt = PromptTemplate(
            template="""You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation.
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  """,
            input_variables=["generation", "documents"],
        )

        self.hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()
    
        # Prompt to check final response is aligned with the user question
        answer_grader_prompt = PromptTemplate(
            template="""You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}""",
            input_variables=["generation", "question"],
        )

        self.answer_grader = answer_grader_prompt | self.llm | JsonOutputParser()


    # Define State
    class GraphState(TypedDict):
        question : str
        generation : str
        company_name: List[str]
        documents : List[str]
    
    # Nodes
    def retrieve(self, state):
        """
        Retrieves documents relevant to the question and filters them based on the company name.

        Args:
            state (dict): The current state containing the query and company names.

        Returns:
            dict: Updated state containing retrieved documents.
        """
        print("---RETRIEVE---")
        question = state["question"]
        company_name = state["company_name"]

        # Retrieval
        filter_dict = [{"company_name": c} for c in company_name]
        if len(filter_dict) > 1:
            documents = self.vector_store.similarity_search(query=question, 
                                                            k=5,
                                                            filter={"$or": filter_dict}
                                                    )
        else:
            documents = self.vector_store.similarity_search(query=question, 
                                                            k=5,
                                                            filter={"company_name": company_name[0]}
                                                    )
        
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """
        Generates a response based on the retrieved documents and the query.

        Args:
            state (dict): The current state containing the query and documents.

        Returns:
            dict: Updated state containing the generated response.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """
        Grades the relevance of retrieved documents and filters irrelevant ones.

        Args:
            state (dict): The current state containing the query and documents.

        Returns:
            dict: Updated state with filtered relevant documents.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def default_reply(self, state):
        """
        Provides a default reply when the system cannot generate a valid response.

        Args:
            state (dict): The current state.

        Returns:
            dict: Default response.
        """
        print("---DEFAULT REPLY---")
        question = state["question"]
        documents = state["documents"]
        #documents = ["Sorry, I cannot answer this question. It is beyond my capability."]
        return {"documents": documents, "question": question, "generation": "Sorry, I cannot answer this question. It is beyond my capability."}

    def grade_generation(self, state):
        """
        Grades the generated response for hallucination and alignment with the query.

        Args:
            state (dict): The current state containing the query, documents, and response.

        Returns:
            str: Grading result indicating whether the response is useful or supported.
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def add_nodes(self):
        """
        Adds nodes to the workflow for different RAG stages.
        """
        self.workflow = StateGraph(self.GraphState)

        # Define the nodes
        self.workflow.add_node("retrieve", self.retrieve) # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents) # grade documents
        self.workflow.add_node("generate", self.generate) # generatae

    def build_graph(self):
        """
        Builds and compiles the workflow graph for managing RAG operations.

        Returns:
            StateGraph: The compiled workflow graph.
        """

        self.add_nodes()
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_edge("grade_documents", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": END,
            },
        )
        memory = MemorySaver()
        app = self.workflow.compile(checkpointer=memory)
        return app

    def execute_graph(self, question, company_name, email):
        """
        Executes the RAG workflow for a given question, company names, and email.

        Args:
            question (str): The user query.
            company_name (list): List of company names to filter the documents.
            email (str): User email for identification in the workflow.

        Returns:
            dict: The final response generated by the workflow.
        """
        entitlement_match = self.entitlement_chain.invoke({"question": question, "company_name": company_name})
        if entitlement_match["match"].lower() == "yes":
            inputs = {"question": question, "company_name": company_name}
            print(email)
            config = {"configurable": {"thread_id": email}}
            app = self.build_graph()
            for output in app.stream(inputs, config):
                for key, value in output.items():
                    # Node
                    print(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                print("\n---\n")
            return value
        else:
            return {"generation": "You're not allowed to access the company report of this company"}
