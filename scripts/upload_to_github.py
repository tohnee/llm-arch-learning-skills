#!/usr/bin/env python3
"""Upload files to GitHub repository using Contents API."""
import base64
import json
import os
import sys
import requests
from pathlib import Path

TOKEN = os.environ.get('GITHUB_TOKEN', '')
USERNAME = os.environ.get('GITHUB_USERNAME', '')

if not TOKEN:
    print("Error: GITHUB_TOKEN environment variable is required")
    sys.exit(1)
if not USERNAME:
    print("Error: GITHUB_USERNAME environment variable is required")
    sys.exit(1)
REPO_NAME = os.environ.get('GITHUB_REPO', 'llm-arch-learning-skills')
REPO_DESC = 'LLM Architecture Learning & Tracking Skill - automated reports for frontier models'

API_BASE = 'https://api.github.com'
HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
}

REPO_ROOT = Path(__file__).parent.parent.resolve()

FILES_TO_UPLOAD = [
    'SKILL.md',
    'README.md',
    '.gitignore',
    'scripts/generate_report.py',
    'scripts/collect.py',
    'scripts/detect_new.py',
    'scripts/upload_to_github.py',
    'assets/report-template.html',
    'references/data-schema.md',
    'references/report-spec.md',
    'references/architecture-concepts.md',
    'references/diagram-cookbook.md',
    'references/source-directory.md',
    'snapshots/2026-06-26.json',
    'tests/__init__.py',
    'tests/test_generate_report.py',
    'reports/index.html',
    'reports/llm-arch-report-2026-06-26-zh.html',
    'reports/llm-arch-report-2026-06-26-en.html',
]

def repo_exists():
    r = requests.get(f'{API_BASE}/repos/{USERNAME}/{REPO_NAME}', headers=HEADERS)
    return r.status_code == 200

def create_repo():
    if repo_exists():
        print(f"Repository {USERNAME}/{REPO_NAME} already exists")
        return
    payload = {
        'name': REPO_NAME,
        'description': REPO_DESC,
        'private': False,
        'auto_init': False,
    }
    r = requests.post(f'{API_BASE}/user/repos', headers=HEADERS, json=payload)
    if r.status_code == 201:
        print(f"Created repository {USERNAME}/{REPO_NAME}")
    else:
        print(f"Failed to create repo: {r.status_code} {r.text}")
        sys.exit(1)

def get_sha(path):
    r = requests.get(f'{API_BASE}/repos/{USERNAME}/{REPO_NAME}/contents/{path}', headers=HEADERS)
    if r.status_code == 200:
        return r.json()['sha']
    return None

def upload_file(rel_path):
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        print(f"  Skipping {rel_path} - not found")
        return False
    with open(full_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('utf-8')
    sha = get_sha(rel_path)
    payload = {
        'message': f'Update {rel_path} (add unit tests)',
        'content': b64,
    }
    if sha:
        payload['sha'] = sha
    r = requests.put(
        f'{API_BASE}/repos/{USERNAME}/{REPO_NAME}/contents/{rel_path}',
        headers=HEADERS,
        json=payload,
    )
    if r.status_code in (200, 201):
        print(f"  ✓ {rel_path}")
        return True
    else:
        print(f"  ✗ {rel_path} failed: {r.status_code} {r.text[:200]}")
        return False

def main():
    print(f"Setting up repository {USERNAME}/{REPO_NAME}...")
    create_repo()
    print(f"\nUploading {len(FILES_TO_UPLOAD)} files...")
    success = 0
    for f in FILES_TO_UPLOAD:
        if upload_file(f):
            success += 1
    print(f"\nDone: {success}/{len(FILES_TO_UPLOAD)} files uploaded")
    print(f"Repository URL: https://github.com/{USERNAME}/{REPO_NAME}")

if __name__ == '__main__':
    main()
