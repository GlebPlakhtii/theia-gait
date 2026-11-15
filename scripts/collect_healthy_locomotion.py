"""Bounded public archive inspection for the healthy locomotion source review.

Zenodo version-pinned records omit ETag. Record this limitation explicitly;
check exact Content-Range, catalog size and member CRC, never claim archive MD5.
"""
import io
import json
import urllib.request
import argparse
import hashlib
import shutil
import zipfile
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/healthy_locomotion_20261114'


class PublicRangeReader(io.RawIOBase):
    def __init__(self, url, size, budget=20_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.pos = self.transferred = 0
        self.cache = b''
        self.cache_start = 0
        self.requests = []
        self.validator = None

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        p = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        if not 0 <= p <= self.size:
            raise ValueError('Seek outside archive')
        self.pos = p
        return p

    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n == 0: return b''
        if self.cache_start <= self.pos and self.pos+n <= self.cache_start+len(self.cache):
            start = self.pos-self.cache_start
            self.pos += n
            return self.cache[start:start+n]
        length = min(self.size-self.pos, max(n, 8192))
        if length > 16_000_000 or self.transferred+length > self.budget:
            raise ValueError('Download budget exceeded')
        start, end = self.pos, self.pos+length-1
        cache_dir = AUDIT/'range_cache'/hashlib.sha256(self.url.encode()).hexdigest()[:12]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir/f'{start}_{end}.bin'
        if cache_file.exists():
            data = cache_file.read_bytes()
            if len(data) != length: raise ValueError('Invalid cached range')
            self.cache, self.cache_start = data, start
            self.pos += n
            return data[:n]
        request = urllib.request.Request(self.url, headers={
            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
            'User-Agent': 'Theia-local-research/0.1'})
        for attempt in range(5):
            time.sleep(.65)  # Stay below the public service request-rate limit.
            try:
                response = urllib.request.urlopen(request, timeout=45)
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 4: raise
                delay = min(60, max(5, int(error.headers.get('Retry-After', '30'))))
                print('Rate limit: waiting', delay, 'seconds', flush=True)
                time.sleep(delay)
        with response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Unexpected HTTP range response')
            validator = response.headers.get('ETag') or response.headers.get('Last-Modified')
            if self.validator and validator != self.validator:
                raise ValueError('Remote entity changed')
            self.validator = validator
            data = response.read(length+1)
            if len(data) != length: raise ValueError('Incorrect byte length')
        self.requests.append([start, end])
        self.transferred += length
        if length <= 8192: cache_file.write_bytes(data)
        if len(self.requests) % 50 == 0:
            print('Archive headers inspected:', len(self.requests), flush=True)
        self.cache, self.cache_start = data, start
        self.pos += n
        return data[:n]


def index_twente():
    import rarfile
    catalog = json.loads((AUDIT/'twente.json').read_text())
    file = catalog['files'][0]
    reader = PublicRangeReader(file['links']['self'], file['size'])
    with rarfile.RarFile(reader) as archive:
        entries = []
        for i in archive.infolist():
            entries.append({'path': i.filename, 'bytes': i.file_size,
                            'compressed_bytes': i.compress_size,
                            'crc32': f'{i.CRC:08x}' if i.CRC is not None else None,
                            'header_offset': i.header_offset,
                            'data_offset': i.data_offset,
                            'compress_type': i.compress_type,
                            'flags': i.flags})
        result = {'source': file, 'entries': entries, 'solid': archive.is_solid(),
                  'http_entity_validator': reader.validator,
                  'bytes_transferred': reader.transferred,
                  'range_requests': reader.requests,
                  'full_archive_checksum_verified': False}
    (AUDIT/'twente_archive_index.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'solid': result['solid'], 'entries': len(entries),
                      'bytes_transferred': reader.transferred, 'sample': entries[:8]}, indent=2))


def acquire_riglet():
    from select_remote_zip import RangeReader, safe_member
    catalog = json.loads((AUDIT/'riglet_article.json').read_text())['files'][0]
    reader = RangeReader(catalog['download_url'], catalog['size'], 250_000_000)
    destroot = ROOT/'data/raw/healthy_locomotion_20261114/riglet'
    subjects = ['AJ026', 'BD004', 'BL025', 'CF027', 'CL007', 'DA013']
    records = []
    with zipfile.ZipFile(reader) as archive:
        selected = [i for i in archive.infolist() if len(i.filename.split('/')) > 3
                    and i.filename.split('/')[1] in subjects
                    and '/Session1/' in i.filename
                    and i.filename.endswith(('Walk_Comfortable.avi', 'Run_Comfortable.avi'))]
        for i in selected:
            safe_member(i)
            parts = i.filename.split('/')
            dest = destroot/parts[1]/parts[2]/parts[-1]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix+'.part')
                with archive.open(i) as src, tmp.open('wb') as out:
                    shutil.copyfileobj(src, out, 1024*1024)
                tmp.rename(dest)
            import zlib
            data = dest.read_bytes()
            assert len(data) == i.file_size and zlib.crc32(data) == i.CRC
            records.append({'source': 'riglet2024', 'subject': parts[1],
                            'session': parts[2], 'activity': 'walk' if 'Walk_' in dest.name else 'run',
                            'environment': 'overground', 'speed_label': 'comfortable',
                            'member': i.filename, 'local_path': str(dest.relative_to(ROOT)),
                            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'crc32_verified': True, 'clinical_form_label': None})
            print('Acquired', parts[1], dest.name, len(data), flush=True)
    (AUDIT/'riglet_acquisition.json').write_text(json.dumps({
        'source': catalog, 'files': records, 'bytes_transferred': reader.transferred,
        'entity_validator': reader.validator, 'range_requests': reader.requests,
        'full_archive_checksum_verified': False}, indent=2))


def acquire_twente():
    import zlib
    index = json.loads((AUDIT/'twente_archive_index.json').read_text())
    assert not index['solid']
    file = index['source']
    reader = PublicRangeReader(file['links']['self'], file['size'], 800_000_000)
    selected = [x for x in index['entries'] if
                any(f'/{subject}/' in x['path'] for subject in ['Subj04','Subj05'])
                and any('/'+trial+'_' in x['path'] for trial in ['walk_45','run_99'])
                and x['path'].endswith('.avi')]
    assert len(selected) == 8
    destroot = ROOT/'data/raw/healthy_locomotion_20261114/twente'
    records = []
    for i in selected:
        # Stored, non-solid, non-encrypted RAR entries can be copied directly.
        assert i['compress_type'] == 48 and i['bytes'] == i['compressed_bytes']
        assert not (i['flags'] & (0x01|0x02|0x04|0x10))
        parts = i['path'].split('/')
        assert len(parts) == 3 and '..' not in parts
        dest = destroot/parts[1]/parts[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            assert shutil.disk_usage(dest.parent).free > i['bytes']+2_000_000_000
            reader.seek(i['data_offset'])
            tmp = dest.with_suffix('.avi.part')
            remaining = i['bytes']
            with tmp.open('wb') as out:
                while remaining:
                    block = reader.read(min(4*1024*1024, remaining))
                    out.write(block);remaining -= len(block)
            tmp.rename(dest)
        sha=hashlib.sha256();crc=0;count=0
        with dest.open('rb') as src:
            while block := src.read(4*1024*1024):
                sha.update(block);crc=zlib.crc32(block,crc);count+=len(block)
        assert count == i['bytes'] and f'{crc:08x}' == i['crc32']
        activity,speed,view,*_ = dest.stem.split('_')
        records.append({'source':'twente2023','subject':parts[1],
            'activity':activity,'environment':'treadmill','speed_kmh':int(speed)/10,
            'view':view, 'trial_group':f'twente2023_{parts[1]}_{activity}_{speed}',
            'member':i['path'],'local_path':str(dest.relative_to(ROOT)),
            'bytes':count,'sha256':sha.hexdigest(),'crc32_verified':True,
            'clinical_form_label':None,'face_anonymization_preserved':True})
        (AUDIT/'twente_acquisition.json').write_text(json.dumps({'source':file,
            'files':records,'bytes_transferred':reader.transferred,
            'range_requests':reader.requests,'http_entity_validator':reader.validator,
            'full_archive_checksum_verified':False},indent=2))
        print('Acquired',parts[1],dest.name,count,flush=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/6644593',timeout=30) as r:
        after=json.load(r)
    assert [(x['key'],x['checksum'],x['size']) for x in after['files']] == [(file['key'],file['checksum'],file['size'])]
    (AUDIT/'twente_catalog_after.json').write_text(json.dumps(after,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['index-twente', 'acquire-riglet', 'acquire-twente'], nargs='?', default='index-twente')
    args = parser.parse_args()
    {'index-twente': index_twente, 'acquire-riglet': acquire_riglet, 'acquire-twente':acquire_twente}[args.action]()
