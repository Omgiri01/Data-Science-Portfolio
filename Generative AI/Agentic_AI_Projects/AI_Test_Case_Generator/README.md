# 🔬 AI Research Assistant Swarm

An advanced, multi-agent research intelligence platform designed to automate deep-dive subject matter investigations. This system utilizes the **Swarm** design pattern to coordinate specialized agents, transforming a simple topic into an executive-grade, fully cited research brief.

---

## 🤖 The Swarm Orchestration

The system delegates steps across four specialized agents in a stateful handoff sequence:

```
           ┌─────────────────────┐
           │     User Query      │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │    Triage Agent     │  ◄── Creates Research Plan
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Query Rewrite Agent │  ◄── Expands into 5-8 search vectors
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   Research Agent    │  ◄── Executed Web Search & Saves Facts
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │    Editor Agent     │  ◄── Compiles final Markdown brief
           └─────────────────────┘
```

1. **Triage Agent**: The "Director" that analyzes the query, generates focus domains, and plans the workflow.
2. **Query Rewriter Agent**: Refines raw inputs into 5–8 diverse and precise search parameters.
3. **Research Agent**: The "Investigator" that scrapes the web and saves important facts.
4. **Editor Agent**: The "Synthesizer" that compiles gathered facts into a cited markdown report.

---

## ✨ Features

- **Autonomous Tooling**: Agents use a custom `save_important_fact` tool to write verified facts into a persistent session store.
- **Interactive UI**: Real-time progress indicators showing current active agent and collected fact statistics.
- **Deep Web Grounding**: Leverages web search tools to gather up-to-date inputs.
- **Structured Outputs**: Implements Pydantic validation schemas to compile reports.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install streamlit openai pydantic
   ```

2. **Configure API Key**:
   Ensure `OPENAI_API_KEY` is exported in your terminal:
   ```bash
   export OPENAI_API_KEY="your_openai_key_here"
   ```

3. **Start Dashboard**:
   ```bash
   streamlit run researchagent.py
   ```
