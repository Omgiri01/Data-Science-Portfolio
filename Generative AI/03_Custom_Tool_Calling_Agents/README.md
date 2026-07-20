# 🛠️ Custom Tool Calling & Agents

This project demonstrates how to build custom mathematical and research tools, bind them to Large Language Models (LLMs), and run sequential agent pipelines.

## 📂 Key Components
- `Agents.py`: Standard agent configurations and reasoning loops.
- `owntool.py`: Custom-built tools configured for agentic invocation.
- `newssummarizer.py`: An agent that fetches news articles and summarizes key highlights.
- `parallelrunnable.py`: Concurrent agent pipeline executions.
- `toolcalling.py`: LangChain tool calling integrations.

## 🚀 Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the tool calling agent:
   ```bash
   python toolcalling.py
   ```
