"""Inspect a public ZIP and fetch only named members using bounded HTTP ranges.

Does not bypass login/access gates. Requires correct 206 responses and unchanged
entity metadata. Member CRC is checked by zipfile; full archive SHA is not verified.
"""
import argparse
import io
import json
import re
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from common import sha256, write_json


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, budget=300_000_000):
        self.url,self.size,self.budget=url,size,budget
        self.pos,self.transferred=0,0
        self.validator=None
        self.requests=[]

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos=(0 if whence==0 else self.pos if whence==1 else self.size)+offset
        if not 0 <= self.pos <= self.size: raise ValueError('Seek outside archive')
        return self.pos

    def read(self, n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if not n:return b''
        if n > 16_000_000 or self.transferred+n > self.budget:
            raise ValueError('Range/budget too large; refuse bulk acquisition')
        start,end=self.pos,self.pos+n-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Theia-research/0.1','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=45) as response:
            expected=f'bytes {start}-{end}/{self.size}'
            if response.status!=206 or response.headers.get('Content-Range')!=expected:
                raise ValueError(f'Incorrect range response: {response.status} {response.headers.get("Content-Range")} expected {expected}')
            validator=response.headers.get('ETag') or response.headers.get('Last-Modified')
            if not validator:raise ValueError('Missing entity validator')
            if self.validator is not None and self.validator!=validator:raise ValueError('Archive changed')
            self.validator=validator
            data=response.read(n+1)
            if len(data)!=n:raise ValueError('Incorrect range length')
        self.requests.append({'start':start,'end':end,'bytes':n})
        self.transferred+=n;self.pos+=n
        return data


def safe_member(info):
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename:
        raise ValueError('Unsafe member path')
    if stat.S_IFMT(info.external_attr>>16) not in (0,stat.S_IFREG,stat.S_IFDIR):
        raise ValueError('Unsafe member type')


def acquire(url,size,output,members=(),directory=None,budget=300_000_000):
    reader=RangeReader(url,size,budget)
    evidence={'url':url,'archive_size':size,'full_archive_sha256_verified':False,'selected_files':[]}
    try:
        with zipfile.ZipFile(reader) as z:
            evidence['entries']=[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,
                                 'crc32':f'{i.CRC:08x}'} for i in z.infolist()]
            for member in members:
                info=z.getinfo(member);safe_member(info)
                if info.file_size>100_000_000:raise ValueError('Member exceeds 100 MB limit')
                if directory is None:raise ValueError('Destination required')
                dest=Path(directory)/PurePosixPath(member).name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():raise ValueError('Refuse overwrite: '+str(dest))
                part=dest.with_name(dest.name+'.part')
                with z.open(info) as src,part.open('xb') as dst:
                    while block:=src.read(1024*1024):dst.write(block)
                part.rename(dest)
                evidence['selected_files'].append({'member':member,'local_path':str(dest.resolve()),
                    'bytes':dest.stat().st_size,'sha256':sha256(dest),'crc32_verified':True})
    except Exception as e:
        evidence['error']=str(e)
        raise
    finally:
        evidence.update(bytes_transferred=reader.transferred,entity_validator=reader.validator,range_requests=reader.requests)
        write_json(output,evidence)
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--size',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--directory')
    p.add_argument('--member',action='append',default=[])
    p.add_argument('--budget',type=int,default=300_000_000)
    a=p.parse_args();r=acquire(a.url,a.size,a.output,a.member,a.directory,a.budget)
    print(f'{len(r["entries"])} entries; {len(r["selected_files"])} files acquired; {r["bytes_transferred"]} bytes transferred')
