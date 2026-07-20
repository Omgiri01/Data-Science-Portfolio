# 💻 SDE Coding Assistant Swarm

A multi-agent software development platform designed to streamline code development—from architectural planning and subtask decomposition to code generation and adversarial code review. Powered by **OpenAI GPT-4o** and the **Swarm** pattern.

---

## 🤖 The Software Development Pipeline

```
           ┌──────────────────────┐
           │   User Requirement   │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │    Planner Agent     │  ◄── Generates tech stack & task trees
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │    Coder Agent       │  ◄── Writes code files for selected tasks
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │   Reviewer Agent     │  ◄── Performs bug & vulnerability scans
           └──────────────────────┘
```

1. **Planner Agent**: Analyzes the project goal, proposes a software architecture, and breaks the project into step-by-step programming tasks.
2. **Coder Agent**: Implements the user's chosen task, writing clean, well-commented modules.
3. **Reviewer Agent**: Conducts adversarial code reviews, scanning for execution bugs, speed bottlenecks, and security vulnerabilities.

---

## ✨ Features

- **Interactive Task Selection**: Allows the user to select specific subtasks to execute.
- **Structured Code Outputs**: Outputs clean modular code blocks alongside configuration templates.
- **Robust Quality Control**: The independent reviewer agent acts as an automated QA gate.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install streamlit openai pydantic
   ```

2. **Configure API Key**:
   Export your OpenAI key:
   ```bash
   export OPENAI_API_KEY="your_openai_key_here"
   ```

3. **Start Dashboard**:
   ```bash
   streamlit run coding_agent.py
   ```
