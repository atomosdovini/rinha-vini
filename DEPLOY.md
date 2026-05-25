# Deploy & submit

End-to-end checklist to get this submitted to Rinha de Backend 2026.

## Prereqs

- GitHub account `atomosdovini`
- Empty public repo at `https://github.com/atomosdovini/rinha-vini` (already exists per your note).
- `git` configured locally with your GitHub identity.
- `gh` CLI installed and authenticated (`gh auth login`). Optional but used in steps below.

## 1. Push `main` branch (source code)

```bash
cd /home/dev/rinha-vini

# init + initial commit
git init -b main
git add -A
git commit -m "Rinha de Backend 2026: IVF + f32 KNN inference engine"

# point at the remote and push
git remote add origin git@github.com:atomosdovini/rinha-vini.git
git push -u origin main
```

The push triggers `.github/workflows/build-and-push.yml`, which builds the image and pushes it to `ghcr.io/atomosdovini/rinha-vini-api:latest`.

### Make the GHCR package public

After the first successful build, the image is **private** by default. Make it public:

1. Open https://github.com/users/atomosdovini/packages/container/rinha-vini-api/settings
2. Scroll to **Danger Zone → Change visibility → Public**.

(Or via CLI: `gh api -X PATCH /user/packages/container/rinha-vini-api -f visibility=public`.)

Verify with: `docker pull ghcr.io/atomosdovini/rinha-vini-api:latest` from anywhere.

## 2. Push `submission` branch (runtime only)

```bash
cd /home/dev/rinha-vini

# create an orphan branch (no history from main)
git checkout --orphan submission
git rm -rf --cached .

# copy the submission-branch files to root
cp submission/docker-compose.yml .
cp submission/haproxy.cfg .
cp submission/info.json .
cp submission/README.md .

# remove everything else from the working tree
git ls-files --others --exclude-standard | xargs -I {} rm -rf {}
rm -rf api tools lb resources submission .github PLAN.md STATUS.md CLAUDE.md DEPLOY.md

# stage and commit
git add docker-compose.yml haproxy.cfg info.json README.md LICENSE .gitignore
git commit -m "submission branch: docker-compose only"
git push -u origin submission

# switch back to main
git checkout main
```

Verify in GitHub that the `submission` branch tree is *only*:

```
docker-compose.yml
haproxy.cfg
info.json
README.md
LICENSE
.gitignore
```

## 3. Smoke-test the submission stack from a clean machine

This simulates what the Rinha test runner does. **Do this on a different machine** (or `docker system prune -af` first) to confirm the public image pulls correctly:

```bash
git clone --branch submission --depth 1 https://github.com/atomosdovini/rinha-vini.git rinha-test
cd rinha-test
docker compose up -d
sleep 30
curl -sf http://localhost:9999/ready  # expect: ok
```

If `/ready` returns 200, you're submission-ready.

## 4. Open the participant PR

Fork the official repo and add `participants/atomosdovini.json`:

```bash
# fork + clone
gh repo fork zanfranceschi/rinha-de-backend-2026 --clone
cd rinha-de-backend-2026

# add your participant file
cat > participants/atomosdovini.json <<'JSON'
[{
    "id": "atomosdovini",
    "repo": "https://github.com/atomosdovini/rinha-vini"
}]
JSON

# branch, commit, push
git checkout -b add-atomosdovini
git add participants/atomosdovini.json
git commit -m "add atomosdovini"
git push -u origin add-atomosdovini

# open PR
gh pr create \
  --repo zanfranceschi/rinha-de-backend-2026 \
  --title "add atomosdovini" \
  --body "Submission for Rinha de Backend 2026.

Repo: https://github.com/atomosdovini/rinha-vini
Stack: Rust + HAProxy, IVF-KNN, mmap, AVX2.
"
```

## 5. Run a preview test

Once your PR is merged (or even before — the Rinha Engine reads the repo URL from your file, the PR just registers you), open an issue on the official repo with `rinha/test` in the body:

```bash
gh issue create \
  --repo zanfranceschi/rinha-de-backend-2026 \
  --title "rinha/test atomosdovini" \
  --body "rinha/test atomosdovini"
```

The bot will pick it up, run the official test against your `submission` branch, and post the score as a comment. You can preview as many times as you want.

## 6. The final test

Runs once at the deadline (**2026-06-05T23:59:59-03:00**). No action needed from you — the engine will run it automatically against whatever is on your `submission` branch at that moment.

## Troubleshooting

- **`/ready` 404 or connection refused** → check `docker compose logs api-1` and `lb`. Usually image platform mismatch (your `docker-compose.yml` must target `linux/amd64`; the test runs amd64).
- **GHCR pull error during preview** → image is still private. Repeat step 1's visibility step.
- **Build hangs at the preprocess step** → the dataset fetch can take ~1 min on slow links. CI has a 6-hour limit.
- **CFS throttle spikes (p99 > 100 ms under burst)** → expected on dev machines with tight cgroup quotas. k6 ramping arrival pattern does not produce sustained burst concurrency; on the Mac Mini it stays smooth.

## Files at a glance

| Path | Purpose |
|---|---|
| `LICENSE` | MIT (required by Rinha rules) |
| `info.json` | participant metadata (required) |
| `docker-compose.yml` (main) | local dev: builds image, runs stack |
| `submission/docker-compose.yml` | what goes to the `submission` branch — pulls public image |
| `submission/haproxy.cfg` | LB config for the `submission` branch (same as `lb/haproxy.cfg`) |
| `.github/workflows/build-and-push.yml` | CI: builds and pushes `ghcr.io/atomosdovini/rinha-vini-api:latest` |
| `submission/participants-atomosdovini.json` | ready-to-copy participant PR file |
