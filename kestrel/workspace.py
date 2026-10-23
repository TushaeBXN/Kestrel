"""A Workspace is one project directory plus its git operations.

GitHub auth: the token is never written into a remote URL or git config. It is
passed per-command as an `http.extraHeader`, and only to github.com remotes.
"""
from __future__ import annotations
import base64, io, os, re, shutil, subprocess, zipfile
from pathlib import Path
from urllib.parse import urlparse

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".next", "dist", "build"}
MAX_UPLOAD = 50 * 1024 * 1024


def kestrel_home() -> Path:
    return Path(os.environ.get("KESTREL_HOME", Path.home() / ".kestrel")).expanduser()


def workspaces_dir() -> Path:
    d = kestrel_home() / "workspaces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9._-]", "-", name.strip()).strip(".-")
    if not n:
        raise ValueError("invalid workspace name")
    return n[:80]


class GitError(RuntimeError):
    pass


class Workspace:
    def __init__(self, root: Path, token_getter=lambda: None):
        self.root = Path(root).resolve()
        self._token = token_getter

    @property
    def name(self) -> str:
        return self.root.name

    # ---- paths -------------------------------------------------------
    def safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    def iter_files(self, base: Path | None = None):
        base = base or self.root
        for dp, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for f in sorted(files):
                yield Path(dp) / f

    def tree(self, limit: int = 2000) -> list[dict]:
        out: list[dict] = []
        for dp, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            rel = Path(dp).relative_to(self.root)
            for d in dirs:
                out.append({"path": str(rel / d), "type": "dir"})
            for f in sorted(files):
                out.append({"path": str(rel / f), "type": "file"})
            if len(out) >= limit:
                break
        return sorted(out[:limit], key=lambda e: e["path"])

    # ---- uploads -----------------------------------------------------
    def save_file(self, rel: str, data: bytes) -> str:
        if len(data) > MAX_UPLOAD:
            raise ValueError("file too large")
        p = self.safe(rel)
        if ".git" in p.relative_to(self.root).parts:
            raise PermissionError("refusing to write inside .git")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p.relative_to(self.root))

    def extract_zip(self, data: bytes) -> int:
        if len(data) > MAX_UPLOAD:
            raise ValueError("zip too large")
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in zf.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        count = 0
        total = 0
        for n in names:
            total += zf.getinfo(n).file_size
            if total > 500 * 1024 * 1024:
                raise ValueError("zip expands too large")
            rel = n.split("/", 1)[1] if strip else n
            if ".git" in Path(rel).parts:
                continue
            self.save_file(rel, zf.read(n))
            count += 1
        return count

    # ---- git ---------------------------------------------------------
    def _auth_args(self, url: str | None) -> list[str]:
        token = self._token()
        if not token or not url:
            return []
        host = urlparse(url).hostname if "://" in url else None
        if host != "github.com":
            return []
        b64 = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        return ["-c", f"http.extraHeader=Authorization: Basic {b64}"]

    def git(self, *args: str, url: str | None = None, timeout: int = 180, cwd: Path | None = None) -> str:
        cmd = ["git", *self._auth_args(url), *args]
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
        try:
            r = subprocess.run(cmd, cwd=cwd or self.root, capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise GitError(f"git {args[0]} timed out")
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip()
            token = self._token()
            if token:
                msg = msg.replace(token, "***")
            raise GitError(msg or f"git {args[0]} failed")
        return r.stdout

    def is_repo(self) -> bool:
        return (self.root / ".git").exists()

    def init(self) -> None:
        if not self.is_repo():
            self.git("init", "-b", "main")

    def branch(self) -> str:
        try:
            return self.git("rev-parse", "--abbrev-ref", "HEAD").strip()
        except GitError:
            return self.git("symbolic-ref", "--short", "HEAD").strip()

    def remote_url(self, remote: str = "origin") -> str | None:
        try:
            return self.git("remote", "get-url", remote).strip()
        except GitError:
            return None

    def status(self) -> dict:
        if not self.is_repo():
            return {"repo": False}
        porcelain = self.git("status", "--porcelain=v1", "-b")
        lines = porcelain.splitlines()
        head = lines[0][3:] if lines else ""
        changes = [{"status": l[:2].strip() or "?", "path": l[3:]} for l in lines[1:]]
        return {"repo": True, "branch": head, "changes": changes,
                "remote": self.remote_url(), "has_token": bool(self._token())}

    def diff(self, max_chars: int = 60000) -> str:
        if not self.is_repo():
            return ""
        try:
            d = self.git("diff", "HEAD")
        except GitError:
            d = self.git("diff")  # no commits yet
        return d[:max_chars] + ("\n... (truncated)" if len(d) > max_chars else "")

    def log(self, n: int = 15) -> list[str]:
        try:
            return self.git("log", f"-{n}", "--oneline").splitlines()
        except GitError:
            return []

    def commit(self, message: str, name: str | None = None, email: str | None = None) -> str:
        if not message.strip():
            raise GitError("empty commit message")
        self.git("add", "-A")
        if not self.git("status", "--porcelain").strip():
            raise GitError("nothing to commit")
        ident: list[str] = []
        for key, val, fallback in (("user.name", name, "Kestrel"), ("user.email", email, "kestrel@localhost")):
            try:
                have = self.git("config", key).strip()
            except GitError:
                have = ""
            if val or not have:
                ident += ["-c", f"{key}={val or fallback}"]
        self.git(*ident, "commit", "-m", message)
        return self.git("rev-parse", "--short", "HEAD").strip()

    def checkout(self, branch: str, create: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.startswith("-"):
            raise GitError("invalid branch name")
        self.git("checkout", *(["-b"] if create else []), branch)

    def set_remote(self, url: str) -> None:
        check_clone_url(url)
        if self.remote_url():
            self.git("remote", "set-url", "origin", url)
        else:
            self.git("remote", "add", "origin", url)

    def push(self, remote: str = "origin", branch: str | None = None) -> str:
        url = self.remote_url(remote)
        if not url:
            raise GitError(f"no remote '{remote}' configured")
        branch = branch or self.branch()
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.startswith("-"):
            raise GitError("invalid branch name")
        out = self.git("push", "-u", remote, branch, url=url)
        return out.strip() or f"pushed {branch} to {remote}"

    def pull(self) -> str:
        url = self.remote_url()
        return self.git("pull", "--ff-only", url=url).strip()


def check_clone_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("https", "ssh") and not re.fullmatch(r"git@[\w.-]+:[\w./-]+", url):
        raise ValueError("only https:// or ssh git URLs are allowed")
    if url.startswith("-"):
        raise ValueError("invalid URL")


def clone(url: str, name: str | None, token_getter=lambda: None) -> Workspace:
    check_clone_url(url)
    base = url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    name = clean_name(name or re.sub(r"\.git$", "", base))
    dest = workspaces_dir() / name
    if dest.exists():
        raise ValueError(f"workspace '{name}' already exists")
    ws = Workspace(dest.parent, token_getter)  # git runs from parent; dest doesn't exist yet
    try:
        ws.git("clone", "--", url, name, url=url, timeout=600)
    except GitError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return Workspace(dest, token_getter)


def create(name: str, token_getter=lambda: None) -> Workspace:
    dest = workspaces_dir() / clean_name(name)
    if dest.exists():
        raise ValueError(f"workspace '{dest.name}' already exists")
    dest.mkdir(parents=True)
    ws = Workspace(dest, token_getter)
    ws.init()
    return ws


def get(name: str, token_getter=lambda: None) -> Workspace:
    dest = workspaces_dir() / clean_name(name)
    if not dest.is_dir():
        raise FileNotFoundError(f"no such workspace: {name}")
    return Workspace(dest, token_getter)


def list_all() -> list[str]:
    return sorted(p.name for p in workspaces_dir().iterdir() if p.is_dir())
