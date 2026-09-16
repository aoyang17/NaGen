"""Bounded result watcher: fetch successful BSCC summary and verify desktop CIFs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=int, required=True)
    p.add_argument('--summary-job', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--desktop', required=True)
    p.add_argument('--timeout-hours', type=float, default=8)
    args = p.parse_args()
    root, desktop = Path(args.out), Path(args.desktop)
    if desktop.exists():
        raise FileExistsError('use a fresh Desktop export folder')
    root.mkdir(parents=True, exist_ok=True)
    status_file = root/'fetch_status.json'
    host = 'scv7eyx@BSCC-N26@ssh.cn-zhongwei-1.paracloud.com'
    remote = f'/data02/home/scv7eyx/projects/NaGen/outputs/full_chgnet_{args.job}'
    deadline = time.monotonic()+args.timeout_hours*3600
    desktop_created = False
    def status(state, **details):
        payload = {'status': state, 'configuration': vars(args), **details}
        status_file.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps(payload), flush=True)
    status('waiting')
    while time.monotonic() < deadline:
        try:
            query = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                f'sacct -X -j {args.summary_job} --format=State -n -P'], capture_output=True,
                text=True, timeout=30, check=True)
            states = [s.strip().split('|')[0] for s in query.stdout.splitlines() if s.strip()]
            if any(s.startswith(('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')) for s in states):
                status('remote_failed', states=states)
                raise SystemExit(2)
            if states and all(s == 'COMPLETED' for s in states):
                subprocess.run(['rsync', '-a', '--partial', f'{host}:{remote}/', str(root)+'/'], check=True, timeout=1200)
                manifest = json.loads((root/'feasible_CIF/manifest.json').read_text())
                for row in manifest['structures']:
                    path = root/'feasible_CIF'/row['file']
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['cif_sha256']:
                        raise ValueError('downloaded CIF hash mismatch')
                if not desktop_created:
                    desktop.mkdir(parents=True, exist_ok=False)
                    desktop_created = True
                subprocess.run(['rsync', '-a', '--ignore-existing', str(root/'feasible_CIF')+'/', str(desktop)+'/'], check=True)
                for source in (root/'feasible_CIF').iterdir():
                    if source.read_bytes() != (desktop/source.name).read_bytes():
                        raise ValueError('Desktop copy verification failed')
                status('completed', exported=len(manifest['structures']))
                return
            status('waiting', states=states)
        except (FileExistsError, ValueError) as error:
            status('verification_failed', error_type=type(error).__name__)
            raise SystemExit(2) from error
        except (subprocess.SubprocessError, OSError) as error:
            status('retrying_connection', error_type=type(error).__name__)
        time.sleep(45)
    status('timed_out')
    raise SystemExit(3)


if __name__ == '__main__':
    main()
