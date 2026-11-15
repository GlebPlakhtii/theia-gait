"""Expand existing public samples without counting overlapping cuts as new trials."""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from acquire_fullbody_samples import ROOT, RAW, AUDIT, CachedChunks, digest, extract_deflate64
from select_remote_zip import RangeReader, safe_member
from collect_healthy_locomotion import PublicRangeReader


def ljmu():
    index=json.loads((AUDIT/'ljmu_rgb_index.json').read_text())
    reader=CachedChunks(index['url'],index['archive_size'],110_000_000)
    entries=[]
    with zipfile.ZipFile(reader) as z:
        for i in z.infolist():
            match=re.fullmatch(r'Participant_(\d+)/Experiment_([123])/E[123]T([12])/P\d+_E[123]T[12]_(?:45|90|Front|Rear)\.mp4',i.filename)
            if match and 2<=int(match[1])<=11:
                safe_member(i)
                entries.append(dict(path=i.filename,header_offset=i.header_offset,bytes=i.file_size,
                    compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}',compression=i.compress_type))
    assert len(entries)==120
    assert reader.validator==index['validator']
    (AUDIT/'ljmu_expansion_index.json').write_text(json.dumps(dict(url=index['url'],validator=reader.validator,entries=entries),indent=2))
    def acquire(e):
        dest=RAW/'ljmu360'/Path(e['path']).name
        for attempt in range(3):
            try:
                r=RangeReader(index['url'],index['archive_size'],e['compressed_bytes']+1_000_000)
                r.validator=index['validator']
                if not dest.exists():extract_deflate64(r,e,dest)
                d=digest(dest)
                assert d['bytes']==e['bytes'] and d['crc32']==e['crc32'],dest.name
                return dict(source='ljmu360',member=e['path'],path=str(dest.relative_to(ROOT)),
                    crc32_verified=True,bytes_transferred=r.transferred,**d)
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(acquire,e):e for e in entries}
        for f in concurrent.futures.as_completed(pending):
            results.append(f.result())
            (AUDIT/'ljmu_expansion_acquisition.json').write_text(json.dumps(dict(url=index['url'],
                entity_validator=index['validator'],full_archive_checksum_verified=False,files=results),indent=2))
            print('LJMU',len(results),'/',len(entries),Path(results[-1]['path']).name,flush=True)


def movi():
    catalog=json.loads((ROOT/'audit/healthy_locomotion_20261114/movi_catalog.json').read_text())
    selected=[x['dataFile'] for x in catalog['data']['latestVersion']['files']
        if re.fullmatch(r'F_PG[12]_Subject_[1345]_L.avi',x['label'])]
    def acquire(f):
        url=f'https://borealisdata.ca/api/access/datafile/{f["id"]}'
        dest=RAW/'movi'/f['filename']
        for attempt in range(3):
            try:
                if not dest.exists():
                    part=dest.with_suffix(dest.suffix+'.part')
                    with urllib.request.urlopen(url,timeout=60) as src,part.open('wb') as out:
                        size=0
                        while b:=src.read(4*1024*1024):
                            size+=len(b)
                            if size>f['filesize']:raise ValueError('Unexpected response length')
                            out.write(b)
                    if size!=f['filesize']:raise ValueError('Incomplete download')
                    part.rename(dest)
                d=digest(dest)
                assert d['md5']==f['md5'] and d['bytes']==f['filesize']
                return dict(source='movi',url=url,path=str(dest.relative_to(ROOT)),md5_verified=True,**d)
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(acquire,selected):
            results.append(result)
            (AUDIT/'movi_expansion_acquisition.json').write_text(json.dumps(dict(files=results,
                scope='Local noncommercial inspection; source terms unchanged'),indent=2))
            print('MoVi',len(results),'/',len(selected),Path(result['path']).name,flush=True)
    index=json.loads((AUDIT/'movi_f_tar_index.json').read_text())
    records=[]
    for subject in [1,3,4,5]:
        entry=next(e for e in index['entries'] if e['name'].endswith(f'/F_v3d_Subject_{subject}.mat'))
        dest=RAW/'movi'/Path(entry['name']).name
        if not dest.exists():
            reader=PublicRangeReader(index['url'],1814554112,entry['size']+1_000_000)
            reader.seek(entry['offset'])
            with dest.with_suffix('.mat.part').open('wb') as out:
                left=entry['size']
                while left:
                    block=reader.read(min(left,4*1024*1024));out.write(block);left-=len(block)
            dest.with_suffix('.mat.part').rename(dest)
        assert dest.stat().st_size==entry['size']
        records.append(dict(member=entry['name'],offset=entry['offset'],path=str(dest.relative_to(ROOT)),**digest(dest)))
        print('MoVi metadata',subject,flush=True)
    (AUDIT/'movi_expansion_metadata.json').write_text(json.dumps(dict(url=index['url'],
        whole_archive_checksum_verified=False,files=records),indent=2))


if __name__=='__main__':{'ljmu':ljmu,'movi':movi}[sys.argv[1]]()
