# app.py — Minimal MCP + Streamlit chat using Groq LLM
# Correct tool message ordering, no filler rendered

import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

# ─────────────────────────────
# Load env (GROQ_API_KEY)
# ─────────────────────────────
load_dotenv()

# ─────────────────────────────
# MCP Servers (CORRECT PATHS)
# ─────────────────────────────
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

SYSTEM_PROMPT = (
    "You have access to tools. "
    "When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

# ─────────────────────────────
# Streamlit UI
# ─────────────────────────────
st.set_page_config(
    page_title="MCP Chat (Groq)",
    page_icon="🧰",
    layout="centered",
)
st.title("🧰 MCP Chat (Groq)")

# ─────────────────────────────
# One-time initialization
# ─────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    st.session_state.client = MultiServerMCPClient(SERVERS)
    tools = asyncio.run(st.session_state.client.get_tools())

    st.session_state.tool_by_name = {t.name: t for t in tools}
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    st.session_state.history = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    st.session_state.initialized = True

# ─────────────────────────────
# Render chat history
# ─────────────────────────────
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ─────────────────────────────
# Chat input (ONLY ONCE)
# ─────────────────────────────
user_text = st.chat_input(
    "Type a message…",
    key="chat_input_main"
)

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    st.session_state.history.append(
        HumanMessage(content=user_text)
    )

    first = asyncio.run(
        st.session_state.llm_with_tools.ainvoke(
            st.session_state.history
        )
    )

    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)

    else:
        st.session_state.history.append(first)

        tool_messages = []
        for tc in tool_calls:
            tool = st.session_state.tool_by_name[tc["name"]]
            args = tc.get("args") or {}
            result = asyncio.run(tool.ainvoke(args))

            tool_messages.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content=json.dumps(result)
                )
            )

        st.session_state.history.extend(tool_messages)

        final = asyncio.run(
            st.session_state.llm.ainvoke(
                st.session_state.history
            )
        )

        with st.chat_message("assistant"):
            st.markdown(final.content or "")

        st.session_state.history.append(
            AIMessage(content=final.content or "")
        )
