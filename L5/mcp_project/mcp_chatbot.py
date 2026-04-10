from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import json
import os

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.openai = OpenAI(api_key=os.getenv("GITHUB_TOKEN"),
                             base_url=os.getenv("GITHUB_ENDPOINT"))
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]

        response = self.openai.chat.completions.create(
            max_tokens=2024,
            model='gpt-4o-mini',
            tools=self.available_tools,
            messages=messages
        )

        process_query_loop = True
        while process_query_loop:
            message = response.choices[0].message

            # Print text content if present
            if message.content:
                print(message.content)

            # Check if response has tool calls
            if message.tool_calls:
                # Process tool calls
                messages.append({'role': 'assistant', 'content': message.content or '', 'tool_calls': message.tool_calls})

                for tool_call in message.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Tool invocation through the client session
                    result = await self.session.call_tool(tool_name, arguments=tool_args)

                    # Append tool result as a separate message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.content[0].text if result.content else str(result)
                    })

                response = self.openai.chat.completions.create(
                    max_tokens=2024,
                    model='gpt-4o-mini',
                    tools=self.available_tools,
                    messages=messages
                )
            else:
                # No more tool calls
                process_query_loop = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    # Initialize the connection
                    await session.initialize()

                    # List available tools
                    response = await session.list_tools()

                    tools = response.tools
                    print("\nConnected to server with tools:", [tool.name for tool in tools])

                    # Convert MCP tools to OpenAI format
                    self.available_tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    } for tool in response.tools]

                    await self.chat_loop()
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            print("Make sure the research_server.py file exists and uv is installed.")


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
