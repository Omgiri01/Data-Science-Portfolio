# Autonomous Multi-Agent AI Research System

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Generative AI & Large Language Models  

---

## 📌 Executive Summary

This project implements an autonomous **Multi-Agent Research Swarm** designed to automate deep web research, web page scraping, report synthesis, and adversarial peer critique. Powered by **LangChain**, **OpenAI GPT-4o**, and **Tavily Web Search APIs**, it coordinates four specialized agents in a multi-stage pipeline to generate comprehensive, cited research papers on any given query.

---

## 🛠️ Multi-Agent Architecture & Pipeline Flow

```
                                  +------------------------------------+
                                  |    User Input Topic / Query        |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | 1. Search Agent (Tavily API)       |
                                  |    - Formulates web queries        |
                                  |    - Extracts top relevant URLs    |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | 2. Reader Agent (Scraper Tool)     |
                                  |    - Scrapes raw web content       |
                                  |    - Cleans HTML to structured text|
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | 3. Writer Agent (Synthesis Chain)  |
                                  |    - Synthesizes findings          |
                                  |    - Formats report with citations |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | 4. Critic Agent (Adversarial Audit)|
                                  |    - Evaluates accuracy & depth    |
                                  |    - Computes quality score /10    |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Final Verified Research Report     |
                                  +------------------------------------+
```

### Agent Roles & Responsibilities:
1. **Search Agent (`build_search_agent`)**: Executes precision web queries via Tavily API to gather relevant article links.
2. **Reader Agent (`build_reader_agent`)**: Fetches raw page content using custom HTTP scraping tools, converting HTML to clean markdown text.
3. **Writer Agent (`writer_chain`)**: Compiles gathered notes into a structured multi-section research report complete with executive summary, key findings, and URL source citations.
4. **Critic Agent (`critic_chain`)**: Conducts an adversarial audit, scoring the report out of 10 and highlighting strengths and areas for improvement.

---

## 📁 Repository Structure

```
04_Multi_Agent_Research_System/
├── app.py                 # Streamlit Web GUI with agent execution logs
├── agents.py              # Agent initialization & chain definitions (Search, Reader, Writer, Critic)
├── pipeline.py            # Multi-agent orchestrator pipeline
├── tools.py               # Web Search (Tavily) & Web Scraping tool functions
├── main.py                # Command-line execution script
└── requirements.txt       # Dependencies manifest
```

---

## 🚀 Getting Started

### 1. Installation
Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project folder:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Execution
Run via Web Dashboard:
```bash
streamlit run app.py
```

Or execute via CLI:
```bash
python main.py
```

---

## 📈 Key Features & Capabilities

- **Autonomous Web Browsing**: Fetches live real-time information from the web.
- **Adversarial Audit Trail**: Built-in Critic Agent scores output quality and flags gaps before delivery.
- **Structured Output**: Automatically formats reports with introduction, key findings, bulleted analysis, and cited source URLs.
