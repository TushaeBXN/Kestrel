# Kestrel

Model-agnostic coding agent in your browser. Clone/upload a repo, let a cheap
local model do the work, review the diff, commit, push.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
kestrel init-models      # writes ~/.kestrel/models.json — edit to add Amy, Anthos, Kerrigan, etc.
kestrel serve            # opens the browser UI (URL includes a per-launch access token)
kestrel run "goal" -p local   # or in a terminal, in the current directory
```

## Local models (cheap) + paid expert (hard tasks only)
Run any GGUF with an OpenAI-compatible server, e.g.
`llama-server -m model.gguf --jinja --port 8080` (`--jinja` is needed for tool calling),
or Ollama / LM Studio / vLLM. Add each as a profile in `~/.kestrel/models.json`.
The `expert` profile (e.g. Claude) is used only when the local model calls
`ask_expert` (you approve each call) or when you click **Review diff with expert**.

## GitHub
Clone any https URL from the sidebar. For private repos / pushing, paste a token
(fine-grained, Contents: read/write) in Settings — kept in server memory only and
sent only to github.com via a per-command header, never stored in the remote URL.

Shell commands, commits, pushes and expert calls ask for approval unless
"auto-approve" is on. The server binds to 127.0.0.1.
