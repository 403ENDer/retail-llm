import getpass
import os
from flask import Flask, request, jsonify, render_template,send_from_directory
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
load_dotenv()
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.prompts import SystemMessagePromptTemplate
from pyprojroot import here
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from flask import Flask, request, render_template, jsonify

from flask import Flask

app = Flask(__name__)

file_path = (
    "retail_sales_dataset.csv"
)

loader = CSVLoader(file_path=file_path)
data = loader.load()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200,separators=["\n\n", "\n", ". ", " ", ""])
splits = []
for doc in data:
    splits.extend(text_splitter.split_documents([doc]))
google_genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=google_genai_embeddings)
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood " 
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt=(
   """ Company Name: Stellar Retail Co.

Overview: Stellar Retail Co. is a leading retail chain specializing in high-quality home goods, fashion, and electronics. Founded in 2010, the company operates both physical stores and an extensive online platform, serving customers across the country.

Mission Statement: To provide exceptional value and unparalleled service in every product we offer, creating a memorable shopping experience for our customers.

answer the question in friendly and professional manner.
if user ask about the sales and products, provide complete result based on the content provided
{context}

Ensure that your response is structured, detailed, and tailored to the needs of the user.
"""
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    store[session_id].messages = store[session_id].messages[-10:]
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_input = data['message']
    response = conversational_rag_chain.invoke({"input": user_input}, config={
        "configurable": {"session_id": "abc123"}
    })
    answer = response.get('answer', 'Sorry, I could not find an answer.')
    formatted_answer = formatAnswer(answer)
    return (formatted_answer)

def formatAnswer(answer):

    formatted_answer = answer.replace('**', '<strong>').replace('</strong>', '</strong>')
    formatted_answer = formatted_answer.replace('## ', '<h2>').replace('\n', '</h2>\n')
    formatted_answer = formatted_answer.replace('--', '<hr>')  
    formatted_answer = formatted_answer.replace('\n', '<br>')
    formatted_answer = formatted_answer.replace('•', '&bull;')
    formatted_answer = formatted_answer.replace('"', '&quot;') 

    return formatted_answer

if __name__ == '__main__':
    app.run(debug=True)
