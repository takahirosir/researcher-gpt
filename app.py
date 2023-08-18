import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv() # load the OPENAI_API_KEY from .env file
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search
'''
use google search api to search for the query, and return the top 5 results
use the api key from https://serpapi.com/
'''

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    }) # query is the input from user, which is the objective & task that user give to the agent

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    } # serper_api_key is the api key from https://serpapi.com/

    response = requests.request("POST", url, headers=headers, data=payload)
    # request the api with the query and api key

    print(response.text)
    # print the search result from Google 

    return response.text

# test
# search("what is the capital of vietnam")
# search("what is meta's thread product")


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request use the browserless api key
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200: # if the request is successful
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# test
# scrape_website("what is langchain", "https://docs.langchain.com/docs/")

def summary(objective, content):
    '''
    method: map reduce, rename summary
    purpose: this summary function is used to summarize the content based on the objective
    why: becaues the GPT3.5 just can handle 4096 tokens
    other ways: vector search
    '''
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    # create a GPT3.5 model

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    # create a text splitter, which will split the content into chunks, each chunk has 10k tokens
    docs = text_splitter.create_documents([content])# split the content into chunks
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    # create a map prompt, which will be used to summarize the content
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    # create a map prompt template, which will be used to summarize the content

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    # load_summarize_chain is a function from langchain, which will load the summarize chain
    # this package have 2 steps: 1. summarize of each chunk 2. combine all the summaries

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    '''
    this class define 2 input that user need to give to the agent: objective & url
    '''
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    '''
    define a function
    '''
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    # define what will be happened when the function is called

    def _arun(self, url: str):
        raise NotImplementedError("error here")
    # define what will be happened when error is raised


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]
# create a list of tools, which will be used to create the agent

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)
# define the system message, which will be passed to the agent, just give the agent a role as a world class researcher
# and give some rules that the agent should follow
# the last two rules are repeated twice, just to make sure the agent will follow it, because this rules is often not followed by the agent  

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
# define the agent kwargs, which will be passed to the agent with system message

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")# base models
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
# the recent conversation will be stored data by data, and the older memory will be summarized

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)
# initialize the agent with the tools, base models, agent type, agent kwargs, memory


# 4. Use streamlit to create a web app
'''
use streamlit to create a web app quickly by the python code
do not forget to pip install streamlit
usage: streamlit run app.py
'''
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")
    # set the page config including title and icon, the page_title is the name of your website

    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
# app = FastAPI()


# class Query(BaseModel):
#     query: str


# @app.post("/")
# def researchAgent(query: Query):
#     query = query.query
#     content = agent({"input": query})
#     actual_content = content['output']
#     return actual_content
