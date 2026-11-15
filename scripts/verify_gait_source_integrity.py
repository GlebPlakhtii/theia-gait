"""Fresh byte-integrity audit of every original video referenced by the corpus."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/prepared/gait_curated_20261114/manifest.json'
OUTPUT=ROOT/'results/gait_metrics_20261114/source_integrity_audit.json'


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    manifest_sha=sha(MANIFEST)
    clips=json.loads(MANIFEST.read_text())['clips']
    grouped=defaultdict(list)
    for clip in clips:grouped[(ROOT/clip['original']).resolve()].append(clip)
    results=[]
    for i,(path,items) in enumerate(sorted(grouped.items()),1):
        result=dict(path=str(path),sources=sorted({c['source'] for c in items}),
                    clips=sorted(c['clip_id'] for c in items))
        try:
            expected={c['source_sha256'] for c in items}
            expected.update(c['original_sha256'] for c in items if c.get('original_sha256'))
            assert len(expected)==1,'Conflicting original-video hashes in curated manifest'
            result['expected_sha256']=next(iter(expected))
            before=path.stat();actual=sha(path);after=path.stat()
            assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during hash verification'
            result.update(bytes=after.st_size,actual_sha256=actual,mtime_ns=after.st_mtime_ns)
            assert actual==result['expected_sha256'],'Original-video bytes differ from curated source provenance'
            result['status']='passed'
        except Exception as error:
            result.update(status='failed',error=type(error).__name__,reason=str(error))
        results.append(result)
        if i%50==0:print(f'Original sources verified: {i}/{len(grouped)}',flush=True)
    assert sha(MANIFEST)==manifest_sha,'Curated manifest changed during source audit'
    counts=dict(Counter(r['status'] for r in results))
    report=dict(updated_at=datetime.now(timezone.utc).isoformat(),
                status='complete' if not counts.get('failed') else 'failed',
                curated_manifest_sha256=manifest_sha,curated_clips=len(clips),
                original_files=len(results),verified_bytes=sum(r.get('bytes',0) for r in results if r['status']=='passed'),
                counts=counts,files=results,
                evidence='Fresh full-file SHA256 of every referenced original video, checked against curated source provenance; stable file identity, size and modification time during each read.',
                limitation='Byte preservation only. Source provenance, consent and clinical validity are not established by matching a hash.')
    temporary=OUTPUT.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(OUTPUT)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','evidence','limitation')}))
    if counts.get('failed'):raise SystemExit(1)


if __name__=='__main__':main()
