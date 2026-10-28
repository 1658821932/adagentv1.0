"""Check staged content without printing matching credentials.

This is a local guard for known credentials and common formats, not a complete
secret detector. Review the staged diff and use team secret scanning as well.
"""
import argparse
import fnmatch
import json
from pathlib import Path
import re
import subprocess


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL)


def configured_secrets(root):
    path=root/'.env'
    if not path.is_file():
        return []
    values=[]
    for line in path.read_text().splitlines():
        match=re.match(r'(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)',line.strip())
        if not match or not any(word in match[1] for word in ('KEY','TOKEN','SECRET','PASSWORD')):
            continue
        value=match[2].strip().strip('"\x27')
        if len(value)>=12:
            values.append(value.encode())
    return values


def check(root):
    names=[p for p in git(root,'diff','--cached','--name-only','--diff-filter=ACMR','-z').decode().split('\0') if p]
    secrets=configured_secrets(root)
    patterns=[
        ('token_like_value',re.compile(rb'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|AKIA[A-Z0-9]{16})')),
        ('private_key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
        ('credential_in_url',re.compile(rb'https?://[^\s/<>"\x27]+:[^\s/<>"\x27]+@')),
    ]
    findings=[]
    for name in names:
        path=Path(name)
        rules=[]
        if any(part in ('.secrets','secrets','.aws') for part in path.parts):
            rules.append('secret_directory')
        if (path.name=='.env' or path.name.startswith('.env.')) and not path.name.endswith('.example'):
            rules.append('local_environment_file')
        if any(name.startswith(p) for p in ('artifacts/','environments/','runtime_libs/','mcp_servers/sentry_video_search/data_nuplan/')):
            rules.append('local_runtime_artifact')
        if any(fnmatch.fnmatch(name,p) for p in ('*.sqlite','*.sqlite3','*.db','*.pth','*.pt','*.ckpt','*.safetensors','*.pem','*.key')):
            rules.append('database_model_or_private_key')
        size=int(git(root,'cat-file','-s',':'+name))
        if size>10*1024**2:
            rules.append('large_staged_file')
        if size<=10*1024**2:
            content=git(root,'show',':'+name)
            if any(secret in content for secret in secrets):
                rules.append('matches_configured_secret')
            rules.extend(rule for rule,pattern in patterns if pattern.search(content))
        if rules:
            display=name
            for secret in secrets:
                display=display.replace(secret.decode(errors='replace'),'[REDACTED]')
            findings.append({'path':display,'rules':rules})
    return {'status':'findings_require_review' if findings else ('passed' if names else 'no_staged_changes'),
        'staged_files_checked':len(names),'findings':findings,
        'scope':'Exact configured local credentials, common token formats, prohibited local paths and large files in the staged snapshot only; no matched values are printed. Historical or unknown secrets need additional review.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    report=check(args.root)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(bool(report['findings']))
