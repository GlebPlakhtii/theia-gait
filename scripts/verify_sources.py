"""Bounded byte probes and resumable downloads. Rights are a separate registry field."""
import argparse
import datetime
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from common import sha256, write_json


def signature(data):
    if data[:4] in (b'PK\x03\x04', b'PK\x07\x08', b'PK\x05\x06'):
        return 'zip'
    if data[4:8] == b'ftyp':
        return 'mp4'
    if data[:4] == b'RIFF' and data[8:12] == b'AVI ':
        return 'avi'
    if data[:4] == b'\x1aE\xdf\xa3':
        return 'matroska_or_webm'
    if data[:4] == b'\x00\x00\x01\xba':
        return 'mpeg_program_stream'
    return 'unknown'


def probe(url):
    result = {'url': url, 'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-4095', 'User-Agent': 'Theia-research/0.1'})
        with urllib.request.urlopen(request, timeout=30) as r:
            data = r.read(4096)
            result.update(http_status=r.status, final_url=r.url, headers=dict(r.headers),
                          bytes_read=len(data), header_hex=data[:32].hex(), signature=signature(data))
            result['access_status'] = 'byte_probe_verified' if signature(data) != 'unknown' else 'unreleased_or_unconfirmed'
    except (urllib.error.URLError, TimeoutError) as e:
        result.update(error=str(e), access_status='unreleased_or_unconfirmed')
    return result


def download(url, dest, expected_bytes, expected_sha256=None):
    """Resume only with a consistent entity validator; never append an ignored Range."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size != expected_bytes:
            raise ValueError('Existing file has wrong length; refusing overwrite')
        digest = sha256(dest)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('Existing checksum mismatch')
        return {'path': str(dest), 'sha256': digest, 'bytes': expected_bytes, 'reused': True}
    part = dest.with_name(dest.name + '.part')
    state = part.with_name(part.name + '.json')
    offset = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent': 'Theia-research/0.1'}
    prior = json.loads(state.read_text()) if state.exists() else {}
    if offset:
        if prior.get('url') != url or not prior.get('validator'):
            raise ValueError('Partial file lacks entity identity; move it aside before retry')
        headers.update(Range=f'bytes={offset}-', **{'If-Range': prior['validator']})
    if offset < expected_bytes:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as r:
            if offset and (r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('Server changed entity or ignored resume; partial preserved')
            validator = r.headers.get('ETag') or r.headers.get('Last-Modified')
            if offset and validator != prior['validator']:
                raise ValueError('Entity validator changed')
            write_json(state, {'url': url, 'validator': validator, 'expected_bytes': expected_bytes})
            with part.open('ab' if offset else 'wb') as f:
                while block := r.read(1024 * 1024):
                    if f.tell() + len(block) > expected_bytes:
                        raise ValueError('Response exceeds declared size')
                    f.write(block)
    if part.stat().st_size != expected_bytes:
        raise ValueError('Incomplete download; partial preserved for resume')
    with part.open('rb') as f:
        if signature(f.read(32)) == 'unknown':
            raise ValueError('Not a recognized media/archive file')
    digest = sha256(part)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Checksum mismatch')
    part.rename(dest)
    return {'path': str(dest), 'sha256': digest, 'bytes': expected_bytes, 'reused': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('url')
    p.add_argument('--output', required=True, help='JSON evidence file')
    p.add_argument('--download', help='Optional local destination')
    p.add_argument('--expected-bytes', type=int)
    p.add_argument('--expected-sha256')
    p.add_argument('--max-bytes', type=int, default=500_000_000)
    a = p.parse_args()
    result = probe(a.url)
    write_json(a.output, result)
    if a.download:
        if not a.expected_bytes or not 0 < a.expected_bytes <= a.max_bytes:
            p.error('Download needs a positive --expected-bytes within --max-bytes')
        if result['access_status'] != 'byte_probe_verified':
            raise ValueError('Probe failed; no download')
        try:
            result['download'] = download(a.url, a.download, a.expected_bytes, a.expected_sha256)
        except Exception as e:
            result['download_error'] = str(e)
            write_json(a.output, result)
            raise
        write_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
