import json, subprocess, os

with open('devquery_bench/repo_list.json') as f:
    repos = json.load(f)['repos']

clone_dir = 'devquery_bench/cloned_repos'
os.makedirs(clone_dir, exist_ok=True)

for i, repo in enumerate(repos, 1):
    name = repo['name']
    url = repo['url']
    safe_name = name.replace('/', '__')
    target = os.path.join(clone_dir, safe_name)
    
    if os.path.exists(target):
        print(f"[{i}/{len(repos)}] ✅ Already cloned: {safe_name}")
        continue
    
    print(f"[{i}/{len(repos)}] 📥 Cloning {name}...")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', url, target],
            check=True, capture_output=True, timeout=300
        )
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n✅ All repos cloned!")