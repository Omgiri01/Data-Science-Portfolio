# 🕸️ LangGraph Agentic Workflows

An advanced project illustrating the implementation of stateful, cyclic AI agent workflows using **LangGraph**. Includes human-in-the-loop validation, conditional RAG, and parallel reducers.

## 📂 Key Workflows
- `conditional_RAG.py`: Integrates routing logic to direct queries to specific vector stores or web search.
- `humanintheloop.py`: Graph checkpointing that pauses execution to wait for manual human approval.
- `iterative_tools.py`: Agent execution patterns that repeatedly call tools until a target criterion is satisfied.
- `parallel_reducers.py`: Processes and reduces states from multiple parallel agent executions.
- `app.py`: A Streamlit dashboard to interactively visualize and run the graph states.

## 🚀 Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the interactive dashboard:
   ```bash
   streamlit run app.py
   ```
