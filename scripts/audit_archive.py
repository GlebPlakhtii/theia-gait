"""List ZIP metadata before optional isolated extraction; never execute contents."""
import argparse
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from common import sha256, write_json


def audit(path, output, destination=None, max_bytes=1_000_000_000):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        write_json(output, {'archive':str(path),'sha256':sha256(path),'entries':[
          {'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,
           'type':'directory' if i.is_dir() else PurePosixPath(i.filename).suffix,
           'mode':oct(i.external_attr>>16)} for i in infos]})
        seen=set()
        if sum(i.file_size for i in infos)>max_bytes:
            raise ValueError('Uncompressed size exceeds budget')
        for i in infos:
            p=PurePosixPath(i.filename);mode=stat.S_IFMT(i.external_attr>>16)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or mode not in (0,stat.S_IFDIR,stat.S_IFREG):
                raise ValueError('Unsafe archive entry: '+i.filename)
            canonical=str(p)
            if canonical in seen:
                raise ValueError('Duplicate path: '+canonical)
            seen.add(canonical)
        if z.testzip():
            raise ValueError('CRC failure')
        if destination:
            root=Path(destination)
            if root.exists():
                raise ValueError('Extraction destination must not exist')
            root.mkdir(parents=True)
            for i in infos:
                dest=root.joinpath(*PurePosixPath(i.filename).parts)
                if i.is_dir():
                    dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(i) as src,dest.open('xb') as dst:
                        shutil.copyfileobj(src,dst)
    return len(infos)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive');p.add_argument('--output',required=True);p.add_argument('--extract-to')
    p.add_argument('--max-bytes',type=int,default=1_000_000_000)
    a=p.parse_args()
    print(audit(a.archive,a.output,a.extract_to,a.max_bytes),'entries verified')
