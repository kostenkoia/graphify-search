## Rules for Token-Efficient Graph Usage
1. ALWAYS call `get_minimal_context` first with a task description.
2. Use `detail_level="minimal"` on all tool calls unless the minimal output is insufficient.
3. Only escalate to `detail_level="standard"` or `"verbose"` for the specific entities that need deeper inspection.
4. Never request more than 3 tool calls per turn unless absolutely necessary.
5. Prefer targeted queries (query_graph with a specific symbol) over broad scans (list_communities with full members).
6. When reviewing changes: detect_changes(detail_level="minimal") → only expand on high-risk items.

## Debug Workflow
1. Call `get_minimal_context(task="debug: <task>")`.
2. Call `semantic_search_nodes(query=<keywords from description>, detail_level="minimal", limit=5)`.
3. For the top 1-2 results, call `query_graph(pattern="callers_of", target=<name>, detail_level="minimal")`.
4. If the issue involves execution flow: call `get_flow(name=<relevant flow>)` for the single most relevant flow.
5. Only call `get_review_context` or `get_impact_radius` if you need to trace the blast radius of a specific change.
