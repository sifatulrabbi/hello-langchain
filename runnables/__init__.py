from dotenv import load_dotenv

load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from icecream import ic


model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

for s in chain.stream({"topic": "bears"}):
    print(s.content, end="", flush=True)

print()
