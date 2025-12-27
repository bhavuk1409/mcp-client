import asyncio
import json
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq

load_dotenv()

SERVERS = { 
    "math": {
        "transport": "stdio",
        "command": "/Users/bhavukagrawal/.local/bin/uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "/Users/bhavukagrawal/Desktop/mcp-tutorial/main.py"
        ]
    },
    "expense": {
        "transport": "streamable_http",
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    },
    "manim-server": {
        "transport": "stdio",
        "command": "/opt/anaconda3/bin/python3",
        "args": [
            "/Users/bhavukagrawal/Desktop/manim-mcp-server/src/manim_server.py"
        ],
        "env": {
            "MANIM_EXECUTABLE": "/opt/anaconda3/bin/manim"
        }
    }
}

async def main():
    # MCP client
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {tool.name: tool for tool in tools}
    print("Available tools:", named_tools.keys())


    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    llm_with_tools = llm.bind_tools(tools)

    prompt = "roll a dice from dice server."

    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("LLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_id = tc["id"]

        result = await named_tools[tool_name].ainvoke(tool_args)
        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_id,
                content=json.dumps(result)
            )
        )

    final_response = await llm_with_tools.ainvoke(
        [prompt, response, *tool_messages]
    )

    print("Final response:", final_response.content)

if __name__ == "__main__":
    asyncio.run(main())
