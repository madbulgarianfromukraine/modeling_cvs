#!/usr/bin/env bash
# setup_git_lfs.sh
# Installs and configures Git LFS tracking for common model files.
# Requires git-lfs to be installed on the system.
set -euo pipefail
# Initialize LFS (idempotent)
git lfs install --local
# Add patterns from .gitattributes
if [ -f .gitattributes ]; then
  git add .gitattributes
  git commit -m "Add .gitattributes for Git LFS" || echo "No changes to commit for .gitattributes"
else
  echo ".gitattributes not found"
fi

echo "Git LFS setup complete. Add and push large files as needed (they will be tracked)."
