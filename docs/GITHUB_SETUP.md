Creating a GitHub repository (local steps)

1. Install GitHub CLI (gh) and authenticate:

```bash
# macOS (brew)
brew install gh
# Linux (deb-based)
sudo apt install gh
# Then login interactively
gh auth login
```

2. Optionally install Git LFS (for large models):

```bash
# Debian/Ubuntu
sudo apt install git-lfs
# then enable
git lfs install --local
```

3. Run the provided helper scripts from the repo root:

```bash
# create a remote repo and push current branch
bash scripts/create_github_repo.sh <github-username>/<repo-name> --private

# setup Git LFS tracking for model files
bash scripts/setup_git_lfs.sh
```

Notes
- `gh repo create` will create the GitHub repository under your account or organization (depending on the repo name you pass) and push the current branch. It requires you to be authenticated via `gh auth login`.
- If you prefer the web UI, create the repo there and then run:

```bash
git remote add origin <REMOTE_URL>
git push -u origin master
```
