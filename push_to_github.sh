#!/bin/bash

echo "=========================================="
echo "🚀 PUSH CODE LÊN GITHUB"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

echo ""
echo "📝 Configuring git user (if needed)..."
read -p "Enter your GitHub username: " github_username
read -p "Enter your email: " github_email

git config user.name "$github_username"
git config user.email "$github_email"
echo "✅ Git user configured"

echo ""
echo "➕ Adding files to git..."
git add .
echo "✅ Files added"

echo ""
echo "💾 Committing changes..."
read -p "Enter commit message (default: 'Backend ready for Render'): " commit_msg
commit_msg=${commit_msg:-"Backend ready for Render"}
git commit -m "$commit_msg"
echo "✅ Changes committed"

echo ""
echo "🌿 Setting main branch..."
git branch -M main
echo "✅ Branch set to main"

echo ""
echo "🔗 Adding remote repository..."
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " repo_url

if git remote get-url origin &> /dev/null; then
    echo "⚠️  Remote 'origin' already exists. Removing..."
    git remote remove origin
fi

git remote add origin "$repo_url"
echo "✅ Remote added"

echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Code pushed to GitHub"
    echo "=========================================="
    echo ""
    echo "📍 Your repository: $repo_url"
    echo ""
    echo "➡️  NEXT STEPS:"
    echo "  1. Go to https://render.com"
    echo "  2. Sign in with GitHub"
    echo "  3. Create New Web Service"
    echo "  4. Connect your repository"
    echo "  5. Deploy!"
    echo ""
else
    echo ""
    echo "❌ Push failed. Check errors above."
    echo ""
    echo "Common issues:"
    echo "  • Authentication: Use Personal Access Token"
    echo "  • Create token at: https://github.com/settings/tokens"
    echo "  • Permissions needed: repo (full control)"
    echo ""
fi
