from mcp.server.fastmcp import FastMCP
import json
from datetime import datetime
import random
import string

mcp = FastMCP("IT Support Server")

# Mock database - in real life this would be Jira's database
tickets_db = {}

@mcp.tool()
def create_ticket(employee_name: str, issue_summary: str, priority: str = "medium") -> str:
    """
    Creates an IT support ticket for an employee.
    
    Args:
        employee_name: Name of the employee raising the ticket
        issue_summary: Brief description of the IT issue
        priority: Priority level - low, medium, or high
    """
    # Generate a random ticket ID like TKT-A3X9
    ticket_id = "TKT-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    
    ticket = {
        "id": ticket_id,
        "employee": employee_name,
        "issue": issue_summary,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assigned_to": "IT Team"
    }
    
    tickets_db[ticket_id] = ticket
    
    return json.dumps({
        "success": True,
        "ticket_id": ticket_id,
        "message": f"Ticket {ticket_id} created successfully",
        "details": ticket
    })

@mcp.tool()
def check_ticket_status(ticket_id: str) -> str:
    """
    Checks the status of an existing IT support ticket.
    
    Args:
        ticket_id: The ticket ID to check (e.g. TKT-A3X9)
    """
    if ticket_id not in tickets_db:
        return json.dumps({
            "success": False,
            "message": f"Ticket {ticket_id} not found"
        })
    
    ticket = tickets_db[ticket_id]
    
    return json.dumps({
        "success": True,
        "ticket_id": ticket_id,
        "status": ticket["status"],
        "issue": ticket["issue"],
        "assigned_to": ticket["assigned_to"],
        "created_at": ticket["created_at"]
    })

@mcp.tool()
def notify_slack(channel: str, message: str, urgency: str = "normal") -> str:
    """
    Sends a notification to a Slack channel to alert the IT team.
    
    Args:
        channel: Slack channel name to send message to (e.g. it-help, incidents)
        message: The message to send to the channel
        urgency: Urgency level - normal or urgent
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    prefix = "🚨 URGENT" if urgency == "urgent" else "ℹ️ INFO"
    
    slack_message = {
        "channel": f"#{channel}",
        "message": f"{prefix}: {message}",
        "sent_at": timestamp,
        "status": "delivered"
    }
    
    print(f"\n[SLACK NOTIFICATION] → #{channel}")
    print(f"  {prefix}: {message}")
    print(f"  Sent at: {timestamp}\n")
    
    return json.dumps({
        "success": True,
        "notification": slack_message
    })

if __name__ == "__main__":
    print("=== IT Support MCP Server ===")
    print("Tools available:")
    print("  - create_ticket")
    print("  - check_ticket_status")
    print("  - notify_slack")
    print("\nServer running...")
    mcp.run()