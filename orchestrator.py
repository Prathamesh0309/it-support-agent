import asyncio
from dotenv import load_dotenv
import os
import json
import re
from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from phase1_rag import retrieve, generate_answer

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Conversation memory - stores the full dialogue history
conversation_history = []

SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            return result.content[0].text
        

async def decide_actions(query: str, rag_answer: str, history: list) -> dict:
    
    # Format history into readable text for the prompt
    history_text = ""
    if history:
        for turn in history:
            role = "Employee" if turn["role"] == "user" else "Copilot"
            history_text += f"{role}: {turn['content']}\n"
    else:
        history_text = "No previous conversation."

    prompt = f"""You are an IT Support Orchestrator Agent.

CONVERSATION HISTORY:
{history_text}

You have already retrieved this answer from the knowledge base:
RAG ANSWER: {rag_answer}

Based on the conversation history and the employee's latest question, decide if any actions need to be taken.
You have access to these tools:
1. create_ticket - Use when the issue needs to be tracked or couldn't be fully resolved
2. notify_slack - Use when the issue is urgent or needs immediate IT attention
3. check_ticket_status - Use when employee is asking about an existing ticket

Respond in this EXACT format and nothing else:
{{
    "needs_ticket": true or false,
    "ticket_priority": "low" or "medium" or "high",
    "needs_slack": true or false,
    "slack_urgency": "normal" or "urgent",
    "reasoning": "one sentence explaining your decision"
}}

EMPLOYEE LATEST QUESTION: {query}"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    raw = response.text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"needs_ticket": False, "needs_slack": False, "reasoning": "Could not parse decision"}


async def orchestrate(query: str, employee_name: str = "Employee") -> str:
    print(f"\n{'='*55}")
    print(f"  Query: {query}")
    print(f"{'='*55}")

    # Step 1: RAG - retrieve and generate grounded answer
    print("\n[1/3] Running RAG pipeline...")
    chunks = retrieve(query)
    rag_answer = generate_answer(query, chunks)
    print(f"  ✓ RAG answer generated from {len(chunks)} chunks")

    # Step 2: Reasoning - decide what actions to take
    print("\n[2/3] Reasoning about actions...")
    decision = await decide_actions(query, rag_answer)
    print(f"  ✓ Decision: {decision['reasoning']}")

    # Step 3: Act - call MCP tools based on decision
    print("\n[3/3] Taking actions...")
    actions_taken = []

    if decision.get("needs_ticket"):
        print("  → Creating ticket...")
        ticket_result = await call_mcp_tool("create_ticket", {
            "employee_name": employee_name,
            "issue_summary": query,
            "priority": decision.get("ticket_priority", "medium")
        })
        actions_taken.append(f"Ticket created: {ticket_result}")
        print(f"  ✓ Ticket created")

    if decision.get("needs_slack"):
        print("  → Notifying Slack...")
        slack_result = await call_mcp_tool("notify_slack", {
            "channel": "it-help",
            "message": f"Employee {employee_name} needs help: {query}",
            "urgency": decision.get("slack_urgency", "normal")
        })
        actions_taken.append(f"Slack notified: {slack_result}")
        print(f"  ✓ Slack notified")

    if not actions_taken:
        print("  ✓ No actions needed - RAG answer is sufficient")

    # Step 4: Compile final response
    final_response = rag_answer
    if actions_taken:
        final_response += "\n\n--- Actions Taken ---"
        for action in actions_taken:
            final_response += f"\n• {action}"

    return final_response


if __name__ == "__main__":
    print("=== IT Support Copilot - Orchestrator ===")
    print("Type 'quit' to exit\n")

    while True:
        employee = input("Your name: ").strip()
        if not employee:
            employee = "Anonymous"

        question = input("Your IT question: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            continue

        answer = asyncio.run(orchestrate(question, employee))
        print(f"\n🤖 IT Copilot:\n{answer}")
        print()