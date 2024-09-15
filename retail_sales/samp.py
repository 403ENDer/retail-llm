import getpass
import os
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

api_key = os.getenv("GROQ_API_KEY")

base_dir = Path(__file__).resolve().parent
db_path = base_dir / 'data' / 'store_details.db'
db=SQLDatabase.from_uri(f"sqlite:///"+str(db_path))

print(db.dialect)
print(db.run("SELECT * from products"))

llm = ChatGroq(model="llama3-8b-8192")

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question by retriving the information from the sql result make sure that your result are correct.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: privide sql result """
)
answer=answer_prompt|llm|StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

a=chain.invoke({"question": "how many products are totally present???"})
print(a)