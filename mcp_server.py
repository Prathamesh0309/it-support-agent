import json
import os
import sys
import uuid
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("IT Support Server")

VALID_PRIORITIES = {"low", "medium", "high"}
# Anchored to this file (not the cwd) so every server process sees the same store.
TICKETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.json")


def log(*args):
    # With the stdio transport, stdout IS the JSON-RPC channel. Anything else
    # printed there corrupts the protocol, so all logging goes to stderr.
    print(*args, file=sys.stderr)


# Mock ticket store. The client starts a fresh server process per tool call, so
# an in-memory dict would forget every ticket immediately. A JSON file stands in
# for Jira's database and survives across processes.
def _load_tickets():
    if not os.path.exists(TICKETS_FILE):
        return {}
    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tickets(tickets):
    tmp = TICKETS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)
    os.replace(tmp, TICKETS_FILE)  # atomic: never leaves a half-written file


@mcp.tool()
def create_ticket(employee_name: str, issue_summary: str, priority: str = "medium") -> str:
    """
    Creates an IT support ticket for an employee.

    Args:
        employee_name: Name of the employee raising the ticket
        issue_summary: Brief description of the IT issue
        priority: Priority level - low, medium, or high
    """
    if priority not in VALID_PRIORITIES:
        priority = "medium"

    tickets = _load_tickets()
    ticket_id = "TKT-" + uuid.uuid4().hex[:6].upper()
    while ticket_id in tickets:  # avoid collisions
        ticket_id = "TKT-" + uuid.uuid4().hex[:6].upper()

    ticket = {
        "id": ticket_id,
        "employee": employee_name,
        "issue": issue_summary,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assigned_to": "IT Team",
    }
    tickets[ticket_id] = ticket
    _save_tickets(tickets)

    return json.dumps({
        "success": True,
        "ticket_id": ticket_id,
        "message": f"Ticket {ticket_id} created successfully",
        "details": ticket,
    })


@mcp.tool()
def check_ticket_status(ticket_id: str) -> str:
    """
    Checks the status of an existing IT support ticket.

    Args:
        ticket_id: The ticket ID to check (e.g. TKT-A3X9F1)
    """
    ticket = _load_tickets().get(ticket_id.strip().upper())
    if ticket is None:
        return json.dumps({"success": False, "message": f"Ticket {ticket_id} not found"})

    return json.dumps({
        "success": True,
        "ticket_id": ticket["id"],
        "status": ticket["status"],
        "issue": ticket["issue"],
        "assigned_to": ticket["assigned_to"],
        "created_at": ticket["created_at"],
    })


@mcp.tool()
def notify_slack(channel: str, message: str, urgency: str = "normal") -> str:
    """
    Sends a notification to a Slack channel to alert the IT team.
    (Mock: logs the message instead of calling the Slack API.)

    Args:
        channel: Slack channel name to send message to (e.g. it-help, incidents)
        message: The message to send to the channel
        urgency: Urgency level - normal or urgent
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "🚨 URGENT" if urgency == "urgent" else "ℹ️ INFO"

    log(f"[SLACK NOTIFICATION] → #{channel}\n  {prefix}: {message}\n  Sent at: {timestamp}")

    return json.dumps({
        "success": True,
        "notification": {
            "channel": f"#{channel}",
            "message": f"{prefix}: {message}",
            "sent_at": timestamp,
            "status": "delivered",
        },
    })


if __name__ == "__main__":
    log("IT Support MCP server running (tools: create_ticket, check_ticket_status, notify_slack)")
    mcp.run()
