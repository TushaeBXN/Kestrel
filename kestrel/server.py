"""Kestrel web server: workspaces, git, uploads, and the streaming agent API.

Because this server can run shell commands, every request except the first
page load must carry the per-launch token (set as an HttpOnly cookie after
visiting the URL printed at startup).
"""
from __future__ import annotations
import asyncio, json, os, secrets, threading
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from . import profiles, workspace as wsmod
from .loop import run as run_loop
from .providers.base import Message
from .workspace import GitError

WEB = Path(__file__).parent / "web" / "index.html"




def create_app(token: str) -> FastAPI:
    app = FastAPI(title="Kestrel")
    state = {"github_token": os.environ.get("GITHUB_TOKEN") or None,
             "git_name": None, "git_email": None}
    histories: dict[str, list[Message]] = {}
    running: dict[str, threading.Event] = {}          # workspace -> cancel flag
    sessions: dict[str, asyncio.Queue] = {}
    pending: dict[tuple[str, str], tuple[threading.Event, dict]] = {}

    def tok():
        return state["github_token"]

    def get_ws(name: str) -> wsmod.Workspace:
        return wsmod.get(name, tok)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path == "/" and request.query_params.get("t") == token:
            resp = RedirectResponse("/")
            resp.set_cookie("kestrel", token, httponly=True, samesite="strict")
            return resp
        if not secrets.compare_digest(request.cookies.get("kestrel", ""), token):
            if request.url.path == "/":
                return HTMLResponse("<h1>Kestrel</h1><p>Open the full URL printed in the terminal "
                                    "(it contains the access token).</p>", status_code=401)
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    def _err(code):
        async def h(request, exc):
            return JSONResponse({"error": str(exc)}, status_code=code)
        return h
    app.add_exception_handler(FileNotFoundError, _err(404))
    for _e in (PermissionError, ValueError, GitError, KeyError):
        app.add_exception_handler(_e, _err(400))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(WEB.read_text(encoding="utf-8"))

    # ---- settings ----------------------------------------------------
    @app.get("/api/config")
    def config():
        cfg = profiles.load()
        return {"profiles": list(cfg["profiles"]), "default_profile": cfg.get("default"),
                "expert": cfg.get("expert"),
                "has_github_token": bool(tok()), "git_name": state["git_name"], "git_email": state["git_email"]}

    @app.post("/api/settings")
    async def settings(req: Request):
        b = await req.json()
        if "github_token" in b:
            state["github_token"] = (b["github_token"] or "").strip() or None
        for k in ("git_name", "git_email"):
            if k in b:
                state[k] = (b[k] or "").strip() or None
        return config()

    # ---- workspaces --------------------------------------------------
    @app.get("/api/workspaces")
    def list_workspaces():
        return {"workspaces": wsmod.list_all()}

    @app.post("/api/workspaces")
    async def new_workspace(req: Request):
        b = await req.json()
        ws = await asyncio.to_thread(wsmod.create, b.get("name", ""), tok)
        return {"name": ws.name}

    @app.post("/api/workspaces/clone")
    async def clone_workspace(req: Request):
        b = await req.json()
        ws = await asyncio.to_thread(wsmod.clone, (b.get("url") or "").strip(), b.get("name"), tok)
        return {"name": ws.name}

    @app.get("/api/workspaces/{name}/tree")
    def tree(name: str):
        return {"tree": get_ws(name).tree()}

    @app.get("/api/workspaces/{name}/file")
    def read_file(name: str, path: str):
        p = get_ws(name).safe(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        data = p.read_bytes()[:500_000]
        return {"path": path, "content": data.decode("utf-8", errors="replace")}

    @app.post("/api/workspaces/{name}/upload")
    async def upload(name: str, request: Request, path: str):
        ws = get_ws(name)
        if int(request.headers.get("content-length", 0)) > wsmod.MAX_UPLOAD:
            return JSONResponse({"error": "file too large"}, status_code=413)
        saved = ws.save_file(path, await request.body())
        return {"saved": saved}

    @app.post("/api/workspaces/{name}/upload-zip")
    async def upload_zip(name: str, request: Request):
        ws = get_ws(name)
        if int(request.headers.get("content-length", 0)) > wsmod.MAX_UPLOAD:
            return JSONResponse({"error": "zip too large"}, status_code=413)
        n = ws.extract_zip(await request.body())
        return {"files": n}

    # ---- git ---------------------------------------------------------
    @app.get("/api/workspaces/{name}/git")
    async def git_info(name: str):
        ws = get_ws(name)
        st = await asyncio.to_thread(ws.status)
        if st.get("repo"):
            st["log"] = await asyncio.to_thread(ws.log)
        return st

    @app.get("/api/workspaces/{name}/diff")
    async def git_diff(name: str):
        return {"diff": await asyncio.to_thread(get_ws(name).diff)}

    @app.post("/api/workspaces/{name}/git/init")
    async def git_init(name: str):
        await asyncio.to_thread(get_ws(name).init)
        return {"ok": True}

    @app.post("/api/workspaces/{name}/git/commit")
    async def git_commit(name: str, req: Request):
        b = await req.json()
        sha = await asyncio.to_thread(get_ws(name).commit, b.get("message", ""),
                                      state["git_name"], state["git_email"])
        return {"commit": sha}

    @app.post("/api/workspaces/{name}/git/branch")
    async def git_branch(name: str, req: Request):
        b = await req.json()
        await asyncio.to_thread(get_ws(name).checkout, b.get("name", ""), bool(b.get("create")))
        return {"ok": True}

    @app.post("/api/workspaces/{name}/git/remote")
    async def git_remote(name: str, req: Request):
        b = await req.json()
        await asyncio.to_thread(get_ws(name).set_remote, (b.get("url") or "").strip())
        return {"ok": True}

    @app.post("/api/workspaces/{name}/git/push")
    async def git_push(name: str):
        out = await asyncio.to_thread(get_ws(name).push)
        return {"result": out}

    @app.post("/api/workspaces/{name}/git/pull")
    async def git_pull(name: str):
        out = await asyncio.to_thread(get_ws(name).pull)
        return {"result": out}

    # ---- agent -------------------------------------------------------
    @app.post("/api/run")
    async def api_run(req: Request):
        b = await req.json()
        goal = (b.get("goal") or "").strip()
        if not goal:
            return JSONResponse({"error": "empty goal"}, status_code=400)
        ws = get_ws(b.get("workspace", ""))
        if ws.name in running:
            return JSONResponse({"error": "this workspace already has a run in progress"}, status_code=409)

        cfg = profiles.load()
        pname = b.get("profile") or cfg.get("default")
        model, provider = profiles.get(pname, cfg)
        expert = profiles.expert_fn(cfg) if pname != cfg.get("expert") else None
        auto = bool(b.get("auto"))
        sid = os.urandom(8).hex()
        q: asyncio.Queue = asyncio.Queue()
        sessions[sid] = q
        cancel = threading.Event()
        running[ws.name] = cancel
        history = histories.setdefault(ws.name, [])
        loop = asyncio.get_running_loop()

        def emit(ev: dict):
            loop.call_soon_threadsafe(q.put_nowait, ev)

        def approve(tc, detail: str) -> bool:
            if auto:
                return True
            ev, box = threading.Event(), {"ok": False}
            pending[(sid, tc.id)] = (ev, box)
            try:
                return ev.wait(600) and box["ok"] and not cancel.is_set()
            finally:
                pending.pop((sid, tc.id), None)

        async def driver():
            try:
                await asyncio.to_thread(run_loop, goal, model, provider, emit, ws, history, approve, cancel, expert)
            except Exception as e:
                emit({"type": "error", "message": str(e)})
            finally:
                running.pop(ws.name, None)
                emit({"type": "stream-end"})

        asyncio.create_task(driver())
        return {"sessionId": sid}

    @app.get("/api/stream/{sid}")
    async def api_stream(sid: str):
        q = sessions.get(sid)
        if not q:
            return JSONResponse({"error": "unknown session"}, status_code=404)

        async def gen() -> AsyncIterator[bytes]:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n".encode()
                if ev.get("type") == "stream-end":
                    sessions.pop(sid, None)
                    break

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/approve")
    async def approve_call(req: Request):
        b = await req.json()
        item = pending.get((b.get("sessionId"), b.get("id")))
        if not item:
            return JSONResponse({"error": "no such pending approval"}, status_code=404)
        item[1]["ok"] = bool(b.get("allow"))
        item[0].set()
        return {"ok": True}

    @app.post("/api/workspaces/{name}/review")
    async def review(name: str):
        """One-shot paid-model code review of the uncommitted diff."""
        ws = get_ws(name)
        diff = await asyncio.to_thread(ws.diff, 40000)
        if not diff.strip():
            return {"review": "No uncommitted changes to review."}
        ask = profiles.expert_fn()
        if not ask:
            raise ValueError("no expert profile configured (see ~/.kestrel/models.json)")
        text = await asyncio.to_thread(ask, "Review this diff for bugs, security issues and missing tests. "
                                            "Be concise; list concrete problems first.\n\n" + diff)
        return {"review": text}

    @app.post("/api/stop/{name}")
    def stop(name: str):
        ev = running.get(name)
        if ev:
            ev.set()
            for (sid, cid), (e, box) in list(pending.items()):
                e.set()  # release any blocked approval so the thread can exit
        return {"stopping": bool(ev)}

    @app.post("/api/reset/{name}")
    def reset(name: str):
        histories.pop(name, None)
        return {"ok": True}

    return app


def serve(host: str, port: int, open_browser: bool = True) -> None:
    import uvicorn
    token = os.environ.get("KESTREL_TOKEN") or secrets.token_urlsafe(24)
    url = f"http://{host}:{port}/?t={token}"
    print(f"Kestrel  v{__import__('kestrel').__version__}")
    cfg = profiles.load()
    print(f"Profiles:  {', '.join(cfg['profiles'])}  (default: {cfg.get('default')}, expert: {cfg.get('expert')})")
    print(f"Workspaces: {wsmod.workspaces_dir()}")
    print(f"\nOpen: {url}\n")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(token), host=host, port=port, log_level="warning")
