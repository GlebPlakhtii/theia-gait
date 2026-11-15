"""Acquire bounded public RGB samples, keeping original frames and provenance.

No login, agreement submission, or restricted-file requests. ZIP members are
CRC-checked; a partial archive never receives a whole-archive checksum claim.
"""
import hashlib
import json
import struct
import subprocess
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

from select_remote_zip import RangeReader, safe_member
from collect_healthy_locomotion import PublicRangeReader

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/fullbody_locomotion_20261114'
RAW = ROOT / 'data/raw/fullbody_locomotion_20261114'


class CachedChunks(RangeReader):
    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n > self.budget-self.transferred:
            raise ValueError('Acquisition budget exceeded')
        if n > 15_000_000:
            cache = AUDIT / f'ljmu_directory_{self.pos}_{n}.bin'
            if cache.exists():
                expected=json.loads((AUDIT/'ljmu_rgb_index.json').read_text())['validator']
                assert self.validator == expected, 'Cached archive directory entity changed'
                data = cache.read_bytes()
                assert len(data) == n
                self.pos += n
                return data
            parts = []
            while n:
                k = min(n, 8_000_000)
                parts.append(super().read(k)); n -= k
                print('Index bytes', self.transferred, flush=True)
            data = b''.join(parts)
            cache.write_bytes(data)
            return data
        return super().read(n)


def digest(path):
    sha, md5, crc, size = hashlib.sha256(), hashlib.md5(), 0, 0
    with path.open('rb') as f:
        while b := f.read(4*1024*1024):
            sha.update(b); md5.update(b); crc = zlib.crc32(b, crc); size += len(b)
    return dict(bytes=size, sha256=sha.hexdigest(), md5=md5.hexdigest(), crc32=f'{crc:08x}')


def extract_deflate64(reader, entry, dest):
    reader.seek(entry['header_offset'])
    fields=struct.unpack('<4s5H3I2H',reader.read(30))
    assert fields[0]==b'PK\x03\x04' and not fields[2]&1
    method=fields[3]
    reader.read(fields[-2]+fields[-1])
    crc=int(entry['crc32'],16);size=entry['bytes'];csize=entry['compressed_bytes']
    nb=dest.name.encode('ascii');mini=dest.with_suffix(dest.suffix+'.member.zip')
    with mini.open('wb') as out:
        out.write(struct.pack('<4s5H3I2H',b'PK\x03\x04',21,0,method,0,0,crc,csize,size,len(nb),0)+nb)
        left=csize
        while left:
            b=reader.read(min(left,4*1024*1024));out.write(b);left-=len(b)
        offset=out.tell()
        central=struct.pack('<4s6H3I5H2I',b'PK\x01\x02',21,21,0,method,0,0,crc,csize,size,len(nb),0,0,0,0,0,0)+nb
        out.write(central);out.write(struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,1,1,len(central),offset,0))
    part=dest.with_suffix(dest.suffix+'.part')
    with part.open('wb') as out:subprocess.run(['unzip','-p',str(mini),dest.name],stdout=out,check=True)
    part.rename(dest);mini.unlink()


def ljmu():
    url = 'https://opendata.ljmu.ac.uk/id/eprint/133/3/Gait%20Dataset.zip'
    reader = CachedChunks(url, 18700445600, 240_000_000)
    records = []
    with zipfile.ZipFile(reader) as z:
        selected = [i for i in z.infolist() if
                    i.filename.startswith('Participant_1/') and
                    i.filename.endswith('.mp4') and
                    any('/E'+str(e)+'T'+str(t)+'/' in i.filename for e in (1,2,3) for t in (1,2))]
        assert 6 <= len(selected) <= 16
        for i in selected:
            safe_member(i)
            dest = RAW/'ljmu360'/Path(i.filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                if i.compress_type==9:
                    extract_deflate64(reader,dict(header_offset=i.header_offset,bytes=i.file_size,
                        compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}'),dest)
                else:
                    part = dest.with_suffix('.mp4.part')
                    with z.open(i) as src, part.open('wb') as out:
                        while b := src.read(1024*1024): out.write(b)
                    part.rename(dest)
            d = digest(dest)
            assert d['bytes'] == i.file_size and d['crc32'] == f'{i.CRC:08x}'
            records.append(dict(source='ljmu360', member=i.filename,
                                path=str(dest.relative_to(ROOT)), crc32_verified=True, **d))
            print('Acquired', dest.name, flush=True)
    (AUDIT/'ljmu_acquisition.json').write_text(json.dumps(dict(url=url,
        entity_validator=reader.validator, bytes_transferred=reader.transferred,
        full_archive_checksum_verified=False, files=records), indent=2))


def lbmc():
    index = json.loads((AUDIT/'lbmc_public_index.json').read_text())
    reader = PublicRangeReader(index['url'], index['size'], 500_000_000)
    selected = [x for x in index['entries'] if '/participant_02/gait/' in x['path'] and x['path'].endswith('.avi')]
    selected += [x for x in index['entries'] if x['path'].endswith('/Calib.toml')]
    records = []
    for entry in selected:
        name = Path(entry['path']).name
        dest = RAW/'lbmc'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            # Build a valid one-member ZIP to use system unzip's Deflate64
            # decoder. The compressed bytes remain exactly as released.
            reader.seek(entry['header_offset'])
            local = reader.read(30)
            fields = struct.unpack('<4s5H3I2H', local)
            assert fields[0] == b'PK\x03\x04' and not (fields[2] & 1)
            method = fields[3]
            assert method == entry['compression']
            reader.read(fields[-2]+fields[-1])
            crc = int(entry['crc32'],16)
            size, csize = entry['bytes'], entry['compressed_bytes']
            nb = name.encode('ascii')
            mini = dest.with_suffix(dest.suffix+'.member.zip')
            with mini.open('wb') as out:
                out.write(struct.pack('<4s5H3I2H',b'PK\x03\x04',21,0,method,0,0,crc,csize,size,len(nb),0)+nb)
                left = csize
                while left:
                    b = reader.read(min(left,4*1024*1024)); out.write(b); left -= len(b)
                offset = out.tell()
                central = struct.pack('<4s6H3I5H2I',b'PK\x01\x02',21,21,0,method,0,0,crc,csize,size,len(nb),0,0,0,0,0,0)+nb
                out.write(central)
                out.write(struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,1,1,len(central),offset,0))
            part = dest.with_suffix(dest.suffix+'.part')
            with part.open('wb') as out:
                subprocess.run(['unzip','-p',str(mini),name], stdout=out, check=True)
            part.rename(dest)
            mini.unlink()
        d = digest(dest)
        assert d['bytes'] == entry['bytes'] and d['crc32'] == entry['crc32']
        records.append(dict(source='lbmc', member=entry['path'],
                            path=str(dest.relative_to(ROOT)),crc32_verified=True,**d))
        (AUDIT/'lbmc_acquisition.json').write_text(json.dumps(dict(url=index['url'],
            scope='Public anonymized preview only; restricted raw videos not requested',
            bytes_transferred=reader.transferred,full_archive_checksum_verified=False,
            files=records),indent=2))
        print('Acquired',name,flush=True)


def movi():
    j = json.loads((ROOT/'audit/healthy_locomotion_20261114/movi_catalog.json').read_text())
    files = j['data']['latestVersion']['files']
    selected = [x['dataFile'] for x in files if x['label'] in [
        'F_CP1_Subject_2.mp4','F_CP2_Subject_2.mp4',
        'F_PG1_Subject_2_L.avi','F_PG2_Subject_2_L.avi']]
    records = []
    for f in selected:
        url = f'https://borealisdata.ca/api/access/datafile/{f["id"]}'
        dest = RAW/'movi'/f['filename']; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            part = dest.with_suffix(dest.suffix+'.part')
            with urllib.request.urlopen(url,timeout=60) as src, part.open('wb') as out:
                size=0
                while b:=src.read(4*1024*1024):
                    size+=len(b)
                    if size>f['filesize']:raise ValueError('Unexpected download length')
                    out.write(b)
            part.rename(dest)
        d = digest(dest)
        assert d['bytes']==f['filesize'] and d['md5']==f['md5']
        records.append(dict(source='movi',url=url,path=str(dest.relative_to(ROOT)),md5_verified=True,**d))
        (AUDIT/'movi_acquisition.json').write_text(json.dumps(dict(files=records,
            scope='Local noncommercial research only; no redistribution'),indent=2))
        print('Acquired',dest.name,flush=True)


if __name__=='__main__':
    {'ljmu':ljmu,'lbmc':lbmc,'movi':movi}[sys.argv[1]]()
