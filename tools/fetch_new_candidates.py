"""Wait for new-candidate selection, fetch artifacts, verify Desktop CIF copies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-job', type=int, required=True)
    p.add_argument('--generation-job', type=int, required=True)
    p.add_argument('--selection-job', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--desktop', required=True)
    args = p.parse_args()
    out, desktop = Path(args.out), Path(args.desktop)
    out.mkdir(parents=True, exist_ok=True)
    if desktop.exists():
        raise FileExistsError('Desktop destination must be fresh')
    host = 'scv7eyx@BSCC-N26@ssh.cn-zhongwei-1.paracloud.com'
    base = '/data02/home/scv7eyx/projects/NaGen/outputs'
    status_path = out/'fetch_status.json'
    deadline = time.monotonic()+8*3600
    desktop_created = False
    def status(state, **details):
        row = {'status': state, 'configuration': vars(args), **details}
        status_path.write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    status('waiting')
    while time.monotonic() < deadline:
        try:
            response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                f'sacct -X -j {args.selection_job} --format=State -n -P'], capture_output=True, text=True,
                timeout=30, check=True)
            states = [line.strip().split('|')[0] for line in response.stdout.splitlines() if line.strip()]
            if any(s.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')) for s in states):
                status('remote_failed', states=states)
                raise SystemExit(2)
            if states and all(s == 'COMPLETED' for s in states):
                subprocess.run(['rsync', '-a', '--partial', f'{host}:{base}/new_candidates_{args.generation_job}/', str(out)+'/'], timeout=1200, check=True)
                subprocess.run(['rsync', '-a', '--partial', f'{host}:{base}/conditional_train_{args.training_job}', str(out)+'/'], timeout=1200, check=True)
                selection = out/'selection'
                report = json.loads((selection/'summary.json').read_text())
                if report['status'] != 'completed':
                    raise ValueError('incomplete selection manifest')
                source = selection/'selected_CIF'
                manifest = json.loads((source/'manifest.json').read_text())
                if len(manifest) != report['selected_count']:
                    raise ValueError('CIF count mismatch')
                for row in manifest:
                    if hashlib.sha256((source/row['file']).read_bytes()).hexdigest() != row['sha256']:
                        raise ValueError('CIF checksum mismatch')
                if not desktop_created:
                    desktop.mkdir(parents=True, exist_ok=False)
                    desktop_created = True
                subprocess.run(['rsync', '-a', '--ignore-existing', str(source)+'/', str(desktop)+'/'], check=True)
                for file in source.iterdir():
                    if file.read_bytes() != (desktop/file.name).read_bytes():
                        raise ValueError('Desktop copy verification failed')
                status('completed', selected_count=len(manifest))
                return
            status('waiting', states=states)
        except (FileExistsError, ValueError) as error:
            status('verification_failed', error_type=type(error).__name__)
            raise SystemExit(2) from error
        except (OSError, subprocess.SubprocessError) as error:
            status('retrying_connection', error_type=type(error).__name__)
        time.sleep(45)
    status('timed_out')
    raise SystemExit(3)


if __name__ == '__main__':
    main()
