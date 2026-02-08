#!/bin/bash
# Setup script for development environment
# Installs git hooks for the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_SOURCE="$PROJECT_ROOT/scripts/git-hooks"
HOOKS_TARGET="$PROJECT_ROOT/.git/hooks"

echo "Setting up development environment..."

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "Error: Not a git repository"
    exit 1
fi

# Install hooks
echo "Installing git hooks..."
for hook in "$HOOKS_SOURCE"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        cp "$hook" "$HOOKS_TARGET/$hook_name"
        chmod +x "$HOOKS_TARGET/$hook_name"
        echo "  ✓ Installed: $hook_name"
    fi
done

echo ""
echo "Development environment setup complete!"
echo ""
echo "Git hooks installed:"
echo "  - pre-push: Validates tag version matches pyproject.toml"
