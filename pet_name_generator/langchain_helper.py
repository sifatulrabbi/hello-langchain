import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("`OPENAI_API_KEY` env is required")
    exit(1)


def generate_pet_name(animal_type: str, pet_color: str):
    llm = OpenAI(temperature=0.7)
    # create a prompt template with langchain prompts
    animal_name_prompt_template = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and it is {pet_color} in color. I want a cool name for it. Suggest me five cool names for my pet.",
    )
    name_chain = LLMChain(
        llm=llm, prompt=animal_name_prompt_template, output_key="pet_name"
    )
    response = name_chain({"animal_type": animal_type, "pet_color": pet_color})
    return response


def langchain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    result = agent.run("What is the average age of a dog? Multiply the age by 3")
    print(result)


if __name__ == "__main__":
    langchain_agent()
    # print(generate_pet_name("Dog", "brown"))
