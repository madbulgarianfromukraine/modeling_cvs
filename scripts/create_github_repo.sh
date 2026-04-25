#!/usr/bin/env bash
# create_github_repo.sh <repo-name> [--private]
# Uses GitHub CLI (`gh`) to create a remote repository and push the current branch.
# Requires: gh CLI installed and `gh auth login` already performed by the user.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <repo-name> [--private]"
  exit 2
fi
REPO_NAME="$1"
PRIVATE_FLAG=""
if [ "${2-}" = "--private" ]; then
  PRIVATE_FLAG="--private"
fi
# Create repo on GitHub
gh repo create "$REPO_NAME" $PRIVATE_FLAG --source=. --remote=origin --push

echo "Remote created and pushed as origin/$REPO_NAME"
