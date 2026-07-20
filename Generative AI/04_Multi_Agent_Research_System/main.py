from agents import AgentState, SupervisorAgent, ResearcherAgent, WriterAgent
from langchain_core.messages import HumanMessage

def run_multi_agent_system():
    print("=== Launching Multi-Agent AI Research System ===")
    
    # Initialize State
    state: AgentState = {
        "messages": [HumanMessage(content="Explain the advantages of multi-agent state graphs.")],
        "next_step": "Supervisor",
        "current_outline": "",
        "gathered_facts": [],
        "final_report": ""
    }

    # Instantiate Agents
    supervisor = SupervisorAgent()
    researcher = ResearcherAgent()
    writer = WriterAgent()

    # Simple Graph Orchestration Loop
    step_count = 0
    max_steps = 10

    while state["next_step"] != "FINISH" and step_count < max_steps:
        print(f"\n--- Step {step_count + 1} | Next Up: {state['next_step']} ---")
        
        if state["next_step"] == "Supervisor":
            decision = supervisor.process(state)
            state["next_step"] = decision["next_step"]
            
        elif state["next_step"] == "Researcher":
            res = researcher.process(state)
            state["messages"].extend(res["messages"])
            state["gathered_facts"].extend(res["gathered_facts"])
            state["next_step"] = res["next_step"]
            
        elif state["next_step"] == "Writer":
            res = writer.process(state)
            state["messages"].extend(res["messages"])
            state["final_report"] = res["final_report"]
            state["next_step"] = res["next_step"]
            
        step_count += 1

    print("\n================================================")
    print("                FINAL REPORT                    ")
    print("================================================")
    print(state["final_report"])

if __name__ == "__main__":
    run_multi_agent_system()
