"""Fetch original ProGait RGB and reference annotations; token stays in memory."""
import concurrent.futures as cf
import getpass
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/progait_20261114'
OUT = ROOT / 'data/raw/ProGait'
BASE = 'https://huggingface.co'
REPO = 'ericyxy98/ProGait'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    token = getpass.getpass('Hugging Face token (hidden): ')
    headers = {'Authorization': 'Bearer ' + token}
    meta = requests.get(f'{BASE}/api/datasets/{REPO}', headers=headers, timeout=60)
    meta.raise_for_status()
    revision = meta.json()['sha']
    entries = []
    url = f'{BASE}/api/datasets/{REPO}/tree/{revision}?recursive=true&expand=false&limit=1000'
    while url:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        entries += r.json()
        url = r.links.get('next', {}).get('url')
        if url and not url.startswith(BASE + '/'):
            raise ValueError('Unexpected pagination host')
    write(AUDIT / 'tree_complete.json', entries)
    selected = [e for e in entries if e['type'] == 'file' and (
        e['path'].startswith('videos/') or e['path'].endswith('_keypoints.npy.gz') or
        e['path'].endswith('.txt') or e['path'] in ['README.md', 'metadata.jsonl', 'LICENSE'])]
    status = dict(status='downloading', revision=revision, repository=REPO,
                  license='cc-by-nc-sa-4.0', selected_files=len(selected),
                  original_videos=sum(e['path'].startswith('videos/') for e in selected),
                  selected_bytes=sum(e['size'] for e in selected), completed=[], failures=[],
                  excluded='Overlay previews, masks and XML exports are not needed for RGB/pose intake.')
    write(AUDIT / 'acquisition_status.json', status)
    print({k:v for k,v in status.items() if k not in ['completed','failures']}, flush=True)

    def download(e):
        relative = Path(e['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe dataset path')
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = e.get('lfs', {}).get('oid')
        if expected_hash and '*' in expected_hash:
            expected_hash = None
        for attempt in range(4):
            try:
                if not target.exists() or target.stat().st_size != e['size']:
                    tmp = target.with_suffix(target.suffix + '.part')
                    with requests.get(f'{BASE}/datasets/{REPO}/resolve/{revision}/{e["path"]}',
                                      headers=headers, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        with tmp.open('wb') as f:
                            for block in response.iter_content(1024 * 1024):
                                f.write(block)
                    if tmp.stat().st_size != e['size']:
                        raise ValueError('Downloaded size mismatch')
                    tmp.replace(target)
                with target.open('rb') as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                if expected_hash and digest != expected_hash:
                    target.rename(target.with_suffix(target.suffix + '.hash_mismatch'))
                    raise ValueError('Downloaded SHA256 mismatch')
                return dict(path=e['path'], bytes=e['size'], sha256=digest,
                            publisher_sha256_verified=bool(expected_hash))
            except Exception as error:
                if attempt == 3:
                    # Never print response URLs (signed URLs can contain credentials).
                    return dict(path=e['path'], error=type(error).__name__)
                time.sleep(2 ** attempt)

    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for index, result in enumerate(pool.map(download, selected), 1):
            status['failures' if 'error' in result else 'completed'].append(result)
            status['updated_at'] = datetime.now(timezone.utc).isoformat()
            write(AUDIT / 'acquisition_status.json', status)
            if index % 25 == 0 or index == len(selected):
                print(f'Downloaded {len(status["completed"])}/{len(selected)}; failures {len(status["failures"])}', flush=True)
    status['status'] = 'failed' if status['failures'] else 'complete'
    write(AUDIT / 'acquisition_status.json', status)
    print('Acquisition', status['status'], flush=True)
    token = None


if __name__ == '__main__':
    main()
