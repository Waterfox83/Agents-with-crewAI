from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
	DirectorySearchTool,
	PDFSearchTool
)
import warnings
import os

from .utils import get_openai_api_key,get_serper_api_key
from langchain.llms import ollama

warnings.filterwarnings('ignore')
serper_api_key = get_serper_api_key()

ollama_llama31= LLM(model="ollama/llama3.1",  base_url="http://localhost:11434")
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()
directory_rag_tool = DirectorySearchTool("/Users/astha1/Documents/Books")
pdf_rag_tool = PDFSearchTool("/Users/astha1/Documents/Books/Cloud-Data-Lakes-For-Dummies-2nd-Snowflake-Special-Edition.pdf")

@CrewBase
class LatestAiDevelopment():
	"""LatestAiDevelopment crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	@agent
	def local_research_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['local_research_analyst'],
			verbose=True,
			# llm=ollama_llama31,
			tools=[pdf_rag_tool]
		)

	@task
	def local_search_task(self) -> Task:
		return Task(
			config=self.tasks_config['local_search_task'],
			output_file='rag_output.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		print(self.agents)
		print(self.tasks)

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
