"""Authenticated, resumable ranged LFS download; verify size and SHA256."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    dest = Path(args.out)
    if dest.exists():
        raise FileExistsError(dest)
    meta = api('repos/aoyang17/NaGen/contents/parent_RDF_opt.jsonl')
    blob = api('repos/aoyang17/NaGen/git/blobs/'+meta['sha'])
    pointer = base64.b64decode(blob['content']).decode()
    expected = re.search(r'oid sha256:([0-9a-f]{64})', pointer).group(1)
    size = int(re.search(r'size (\d+)', pointer).group(1))
    assert size == meta['size']
    parts = dest.parent/(dest.name+'.chunks')
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 4*1024*1024
    def fetch(index):
        start = index*chunk_size
        end = min(size, start+chunk_size)-1
        path = parts/f'{index:05d}.part'
        if path.exists() and path.stat().st_size == end-start+1:
            return index
        for attempt in range(5):
            try:
                request = urllib.request.Request(meta['download_url'],
                    headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('server did not honor exact byte range')
                    with path.open('wb') as handle:
                        while block := response.read(256*1024):
                            handle.write(block)
                if path.stat().st_size != end-start+1:
                    raise ValueError('incomplete chunk')
                return index
            except Exception:
                if attempt == 4:
                    # Do not expose signed URLs through exception strings.
                    raise RuntimeError(f'chunk {index} failed after 5 attempts') from None
                time.sleep(2*(attempt+1))
    count = (size+chunk_size-1)//chunk_size
    print(json.dumps({'status': 'downloading', 'bytes': size, 'sha256': expected,
                      'chunks': count, 'workers': args.workers}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
            print(f'completed {done}/{count}: chunk {future.result()}', flush=True)
    temporary = dest.with_suffix(dest.suffix+'.assembling')
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for i in range(count):
            with (parts/f'{i:05d}.part').open('rb') as handle:
                while block := handle.read(1024*1024):
                    output.write(block)
                    digest.update(block)
    if temporary.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError('full file integrity check FAILED; not publishing dataset')
    if dest.exists():
        raise FileExistsError(dest)
    temporary.rename(dest)
    print(json.dumps({'status': 'verified', 'path': str(dest), 'bytes': size,
                      'sha256': expected}), flush=True)


if __name__ == '__main__':
    main()
