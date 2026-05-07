# Deploying kb-core to a remote server

Targets a single-tenant Linux box (Ubuntu 22.04+ assumed; works on CentOS too).
For a personal-use deployment the recommended access pattern is **SSH tunnel** —
the web UI binds to `127.0.0.1` on the server and you tunnel to it from your
laptop. Streamlit has no built-in auth; do **not** expose port 8501 directly to
the public internet without putting nginx + basic-auth (or similar) in front.

There are two deployment paths:

- **Path A (Docker, recommended for Aliyun mainland)** — build the image
  locally with bge-m3 baked in, transfer as a tarball, `docker load + run`.
  Avoids the HuggingFace Hub flakiness on Aliyun mainland and gives you a
  known-good artifact. See section [A](#path-a-docker-build-locally-ship-tarball).
- **Path B (git clone + uv sync, original)** — server pulls the source and
  installs deps. Faster initial cycle if HF and GitHub are both reachable.
  Original 7-step recipe below.

---

## Path A — Docker (build locally, ship tarball)

### A.1 Prerequisites on your laptop

Install Docker. On macOS we recommend **OrbStack** (faster, lower RAM than
Docker Desktop, free for personal use):

    brew install orbstack

Or Docker Desktop:

    brew install --cask docker

Then start it once and confirm:

    docker version

### A.2 Build the image

From the repo root:

    ./scripts/docker-build.sh

This:
1. `docker build` — installs Python 3.12 + `uv sync` + bakes in the bge-m3
   embedding model (~4 GB, so the model is fully offline-ready).
2. `docker save` → `kb-core-latest.tar` (~3 GB compressed).

You can tag a version:

    ./scripts/docker-build.sh v1.0.0     # produces kb-core-v1.0.0.tar

### A.3 Transfer to the server

    rsync -avz --progress kb-core-latest.tar root@your-server:/root/

    # And the project data (PDFs + structured/ + db/):
    rsync -avz --progress \
        data/projects/ \
        root@your-server:/root/pichia-kb/data/projects/

### A.4 Server-side: configure secrets

    ssh root@your-server
    mkdir -p /root/pichia-kb
    # Build a .env file. On Aliyun mainland the critical line is
    # KB_DEFAULT_MODEL=deepseek-chat (Gemini API is unreachable).
    cat > /root/pichia-kb/.env <<'EOF'
    DEEPSEEK_API_KEY=sk-...
    KB_DEFAULT_MODEL=deepseek-chat
    EOF
    chmod 600 /root/pichia-kb/.env

### A.5 Load + run

    docker load -i /root/kb-core-latest.tar      # imports the image
    docker images | grep kb-core                  # confirm it's there

    docker run -d \
        --name kb-core \
        --restart unless-stopped \
        -p 127.0.0.1:8501:8501 \
        --env-file /root/pichia-kb/.env \
        -v /root/pichia-kb/data:/app/data \
        kb-core:latest

Logs:

    docker logs -f kb-core

### A.6 Access from your laptop (SSH tunnel)

    # On your laptop:
    ssh -L 8501:localhost:8501 root@your-server
    # Browser: http://localhost:8501

### A.7 Update later

When you push code changes:

    # Locally:
    git push
    ./scripts/docker-build.sh v1.0.1
    rsync -avz kb-core-v1.0.1.tar root@your-server:/root/

    # On the server:
    docker load -i /root/kb-core-v1.0.1.tar
    docker stop kb-core && docker rm kb-core
    docker run -d --name kb-core --restart unless-stopped \
        -p 127.0.0.1:8501:8501 \
        --env-file /root/pichia-kb/.env \
        -v /root/pichia-kb/data:/app/data \
        kb-core:v1.0.1

### Known limits on Aliyun mainland (Docker path)

| Feature | Works? | Why |
|---|---|---|
| 💬 Q&A | ✅ via DeepSeek | `KB_DEFAULT_MODEL=deepseek-chat` |
| Browse pages (2/3/4/5/6) | ✅ | Read-only, no LLM needed |
| 🔍 Vector search | ✅ | bge-m3 baked into image |
| 🛠️ Schema Curator | ❌ | Gemini function-calling only |
| `kb extract-figures` | ❌ | Vision still Gemini-only by default |

---

## Path B — git clone + uv sync (original)

## 0. Prerequisites on the server

```bash
ssh root@your-server         # or your sudo user
# Install uv (Python package manager) if not already
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart shell
uv --version                 # confirm
```

`uv` will manage the Python 3.12 install — no system python required.

---

## 1. Clone repo + install deps

```bash
cd ~                         # or wherever you want it
git clone https://github.com/trytobebee/pichia-kb.git
cd pichia-kb
uv sync                      # creates .venv, installs everything
```

---

## 2. Configure secrets

```bash
cp .env.example .env
nano .env                    # set GEMINI_API_KEY=...
chmod 600 .env               # owner-only read
```

---

## 3. Transfer project data

The `data/projects/<slug>/` directory contains your PDFs + extracted JSONs +
ChromaDB. It is gitignored, so you have to copy it from your laptop.

```bash
# From your LOCAL machine (not the server):
rsync -avz --progress \
    data/projects/ \
    root@your-server:/root/pichia-kb/data/projects/
```

For pichia-collagen the payload is roughly 95 MB (papers 32M + figures 44M +
db 15M + structured 4M).

Verify on the server:

```bash
~/pichia-kb/.venv/bin/kb list-projects --help    # confirms install ok
~/pichia-kb/.venv/bin/kb status --project pichia-collagen
```

---

## 4. Run it

### Option A: ad-hoc with tmux (quickest)

```bash
cd ~/pichia-kb
tmux new -s kb               # new tmux session
./scripts/start.sh           # launches streamlit on :8501
# Ctrl-B then D to detach. The server keeps running.
# tmux attach -t kb          # to come back later
```

### Option B: systemd (auto-restart, starts on boot)

```bash
# Edit the unit template if your install path isn't /root/pichia-kb
sudo cp ~/pichia-kb/scripts/kb-core.service /etc/systemd/system/kb-core.service
sudo nano /etc/systemd/system/kb-core.service   # confirm User / WorkingDirectory / EnvironmentFile

sudo systemctl daemon-reload
sudo systemctl enable --now kb-core
sudo systemctl status kb-core               # should be "active (running)"
sudo journalctl -u kb-core -f               # live logs
```

To upgrade later:
```bash
cd ~/pichia-kb && git pull && uv sync && sudo systemctl restart kb-core
```

---

## 5. Access the web UI from your laptop (SSH tunnel)

```bash
# On your laptop:
ssh -L 8501:localhost:8501 root@your-server

# Keep that SSH session open. In your browser:
#   http://localhost:8501
# Traffic is encrypted by SSH; nothing exposed publicly.
```

Tip: add a Host alias to `~/.ssh/config` so you can just type `ssh kb-server`
and forget the tunnel flag:

```
Host kb-server
    HostName your-server-ip
    User root
    LocalForward 8501 localhost:8501
```

---

## 6. (Optional) Public access with nginx + basic auth

Only if you have non-technical collaborators who can't use SSH. Skip if you
don't need it.

```bash
sudo apt install nginx apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd alice          # set username/password
# create /etc/nginx/sites-available/kb-core
```

```nginx
server {
    listen 80;
    server_name kb.your-domain.com;

    auth_basic "kb-core";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/kb-core /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
# Open port 80 (and 443 if using TLS) in your Aliyun security group.
# Strongly recommend adding TLS via Let's Encrypt: `sudo certbot --nginx -d kb.your-domain.com`
```

The streamlit unit still binds to 127.0.0.1 — only nginx talks to the outside.

---

## 7. Backups

The local data is the asset, not the code. At minimum periodically rsync
`data/projects/` back to your laptop:

```bash
# From your LOCAL machine
rsync -avz --progress \
    root@your-server:/root/pichia-kb/data/projects/ \
    backups/projects-$(date +%F)/
```

ChromaDB is a normal SQLite-backed file store — copying the directory while
the service is stopped (or with `chromadb`'s built-in snapshot) is safe.

---

## Common issues

**`ModuleNotFoundError: No module named 'kb_core'`** — you're running outside
the venv. Use `./scripts/start.sh` or `source .venv/bin/activate` first.

**Streamlit page loads but is empty / project picker missing** — your `data/projects/`
didn't sync over. Check `ls data/projects/` on the server.

**`KeyError: 'GEMINI_API_KEY'`** — `.env` not loaded. systemd uses
`EnvironmentFile=`; manual launch needs `set -a; . .env; set +a` (the
start.sh does this).

**Port 8501 already in use** — another streamlit is running. Either reuse
that one or kill it: `pkill -f "streamlit run"`.

**Gemini API unreachable from Aliyun mainland** — `curl
https://generativelanguage.googleapis.com` will hang or fail. The fix is
NOT a proxy; it's switching providers. The framework supports DeepSeek
out of the box (OpenAI-compatible API, accessible from mainland China):

1. Get a DeepSeek API key from <https://platform.deepseek.com/>
2. Add it to your server's `.env`:
   ```
   DEEPSEEK_API_KEY=sk-...
   ```
3. Use a DeepSeek model id in CLI calls (or set as default in your
   workflow scripts):
   ```bash
   kb ingest paper.pdf --project foo --model deepseek-chat
   kb ask "..." --project foo --model deepseek-chat
   kb review --project foo --model deepseek-chat
   ```
4. Limitations: `kb extract-figures` (vision) and the 🛠️ Schema Curator
   web page still need Gemini. If you must keep both on the same server
   without Google access, run figure extraction locally and rsync the
   output `data/projects/<slug>/structured/figures/` up.
