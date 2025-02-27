import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from utils import get_user_input
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def main():
    """Main function to run the RAG application."""
    # 1. Collect user input
    web_page_url= get_user_input()
    embeddings = setup_llm_and_embeddings()
    # 2. Load and process the web page content
    db = load_and_process_web_page(web_page_url, embeddings)
    print('Vector stor created successfully.')

    # 4. Create the RAG chain
    # ... .create_rag_chain()

    # 5. Run the chain and display the answer
    # ... .run_chain_and_display_answer(question)

# Function to load and process the web page content
def load_and_process_web_page(web_page_url, embeddings):
    """Load the web page content, split it into chunks, and embed them."""
    loader = WebBaseLoader(web_page_url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    db = Chroma.from_documents(docs, embedding=embeddings)
    return db

# Function to set up the LLM and embeddings
def setup_llm_and_embeddings():
    """Initialize the Google Gemini LLM and embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    return embeddings


# Function to set up the LLM and embeddings
def setup_llm():
    """Initialize the Google Gemini LLM and embeddings."""
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    return model

# Function to create the RAG chain
def create_rag_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to 
    provide  all the details, if the answer is not in provided context just say, "answer is not
    available in the context", do not provide the wrong answer\n\n
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff",
                          prompt=prompt)
    return chain

# Function to run the chain and display the answer
def run_chain_and_display_answer(question):
    """Run the RAG chain with the given question and display the answer."""
    # ... Implementation to be added
    pass  # Placeholder for implementation

if __name__ == "__main__":
    main()
