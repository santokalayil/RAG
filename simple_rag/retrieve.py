import os
import dotenv
from textwrap import dedent
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field
from IPython.display import Markdown, display
import lancedb

from helpers import EmbeddingGenerator, DB_URI


from nest_asyncio import apply
apply()

class Output(BaseModel):
    answer_in_markdown: str

    def show_md(self) -> None:
        display(Markdown(self.answer_in_markdown))

# System and user prompt templates
SYS_PROMPT = dedent("""
You are a helpful assistant. Give a descriptive answer to the user's question.
""")
USR_PROMPT_TEMPLATE = dedent("""
Question: {query}

Carefully study the following context to answer user's question:

{context}
""")

# Initialize the model and agent
model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model, system_prompt=SYS_PROMPT, output_type=Output)

# Connect to LanceDB
table_name = "ai_library_documentation"
db = lancedb.connect(DB_URI)

db.table_names()
# Open the table
tbl = db.open_table(table_name)

tbl.to_pandas()

# query = "How to create basic pydantic ai agent?"
# query = "why pydantic graph is better?"
query = "How to convert a pydantic ai agent to MCP? I need webapi and not stdio"

# Generate embedding for the query
embed = EmbeddingGenerator()
qembed = embed.generate(query)

# Perform vector search
top_n = 30
search_results = tbl.search(qembed.vector, vector_column_name="vector").select(["content"]).limit(top_n).to_pandas()

# Prepare the context for the query
line = 100 * "="
search_results_str = f"\n{line}\n".join(search_results["content"].to_list())
query_with_context = USR_PROMPT_TEMPLATE.format(query=query, context=search_results_str)

# Run the agent with the query and context
res = await agent.run(query_with_context)

# Display the result
res.output.show_md()

