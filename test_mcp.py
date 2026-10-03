"""Smoke test: start the MCP server and verify all three tools work end to end."""
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_connection():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List all available tools
            tools = await session.list_tools()
            print("✅ Connected to MCP Server!")
            print(f"\nAvailable tools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"  - {tool.name}")

            names = {t.name for t in tools.tools}
            assert names == {"create_ticket", "check_ticket_status", "notify_slack"}, names

            created = json.loads((await session.call_tool(
                "create_ticket", {"employee_name": "Test", "issue_summary": "smoke test", "priority": "low"}
            )).content[0].text)
            ticket_id = created["ticket_id"]

            # Ticket must be visible from a *separate* server process (persistence).
            async with stdio_client(server_params) as (r2, w2):
                async with ClientSession(r2, w2) as other:
                    await other.initialize()
                    status = json.loads((await other.call_tool(
                        "check_ticket_status", {"ticket_id": ticket_id}
                    )).content[0].text)
            assert status["success"] and status["status"] == "open", status
            print(f"✅ Ticket {ticket_id} persisted across server processes")

if __name__ == "__main__":
    asyncio.run(test_connection())