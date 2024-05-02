from flask import Flask, request, render_template_string
import os
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Set up the Flask application
app = Flask(__name__)



# Set up Chroma database
CHROMA_PATH = "chroma"
loader = CSVLoader("oscar_text.csv", encoding="utf-8")
documents = loader.load()
db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Set up LangChain components
template = """You are a helpful AI assistant and your goal is to answer questions as accurately as possible based on the context provided. Be concise and just include the response:

context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"]
        context = retriever.get_relevant_documents(user_query)
        answer = chain.invoke(user_query)
        return render_template_string(HTML_TEMPLATE, query=user_query, answer=answer, context=context)
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System</title>
</head>
<body>
    <h1>RAG System Query Interface</h1>
    <form action="/" method="post">
        <label for="query">Enter your query:</label>
        <input type="text" id="query" name="query">
        <button type="submit">Submit</button>
    </form>
    {% if context %}
        <h2>Generated Context:</h2>
        <p>{{ context }}</p>
    {% endif %}
    {% if answer %}
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
