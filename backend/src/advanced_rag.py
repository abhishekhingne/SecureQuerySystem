#Importing the python dependencies
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class AdvancedRAG:
    def __init__(self, llm_url, vector_store) -> None:
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
            Use three sentences maximum and keep the answer concise. Make sure Answer is in markdown format and do not mention anything about the figures, charts or any visualisation.
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
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
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
        print("---DEFAULT REPLY---")
        question = state["question"]
        documents = state["documents"]
        #documents = ["Sorry, I cannot answer this question. It is beyond my capability."]
        return {"documents": documents, "question": question, "generation": "Sorry, I cannot answer this question. It is beyond my capability."}

    def grade_generation(self, state):
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
        self.workflow = StateGraph(self.GraphState)

        # Define the nodes
        self.workflow.add_node("retrieve", self.retrieve) # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents) # grade documents
        self.workflow.add_node("generate", self.generate) # generatae

    def build_graph(self):
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
        app = self.workflow.compile()
        return app

    def execute_graph(self, question, company_name):
        entitlement_match = self.entitlement_chain.invoke({"question": question, "company_name": company_name})
        if entitlement_match["match"].lower() == "yes":
            inputs = {"question": question, "company_name": company_name}
            app = self.build_graph()
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node
                    print(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                print("\n---\n")
            return value
        else:
            return "You're not allowed to access the company report of this {} company".format(", ".join(company_name))
