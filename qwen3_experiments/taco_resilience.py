"""Compute-local storage recovery and bounded GRPO concurrency fallback."""

from datetime import datetime
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from qwen3_experiments.taco_eval import digest, now, read, write

SPACE_ERRORS = (errno.ENOSPC, errno.EDQUOT)


def log_tail(path, limit=2 * 1024**2):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def cuda_oom(text):
    return bool(re.search(
        r'CUDA(?: error)?[^\n]{0,80}out of memory|'
        r'(?:torch\.)?(?:cuda\.)?OutOfMemoryError:[^\n]*(?:CUDA|GPU)|'
        r'(?:ValueError|RuntimeError): No available memory for the cache blocks', text, re.IGNORECASE))


def fallback_concurrency(initial, previous, saved):
    if saved.get('max_num_seqs') == 16:
        return 16
    if initial == 32 and previous.get('exit_code') not in (None, 0):
        if previous.get('cuda_oom') or cuda_oom(log_tail(previous.get('log', '/dev/null'))):
            return 16
    return initial


def local_log(plan, name):
    local = Path(plan['scratch']) / 'logs' / name
    local.parent.mkdir(parents=True, exist_ok=True)
    source = Path(plan['output_root']) / name
    try:
        if source.is_symlink():
            if source.resolve() != local:
                raise ValueError('Unexpected log redirect')
        elif not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            source.symlink_to(local)
    except OSError as exc:
        if exc.errno not in SPACE_ERRORS:
            raise
    return local


def process(pid):
    folder = Path('/proc') / str(pid)
    try:
        fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'uid': folder.stat().st_uid, 'start_ticks': fields[19],
                'args': (folder / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'),
                'cgroup': (folder / 'cgroup').read_text()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def processes():
    return [p for entry in Path('/proc').glob('[0-9]*')
            if (p := process(entry.name)) and p['uid'] == os.getuid()]


def terminate_failed_step(plan, record):
    """Only a recorded failed training step, never the allocation or another run."""
    step, group = str(record.get('slurm_step_id', '')), record.get('cgroup', '')
    if not step.isdigit() or f'/step_{step}/' not in group:
        return []
    command = ['scontrol', 'listpids', plan['job_id']]
    output = subprocess.check_output(command, text=True, timeout=15)
    owned = {int(parts[0]) for line in output.splitlines()[1:]
             if len(parts := line.split()) >= 3 and parts[0].isdigit()
             and parts[1:3] == [plan['job_id'], step]}
    victims = [p for p in processes() if p['pid'] in owned and p['cgroup'] == group and p['pid'] != os.getpid()]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = [p for p in victims if process(p['pid']) == p]
        for p in live:
            try:
                os.kill(p['pid'], sig)
            except ProcessLookupError:
                pass
        if not live:
            break
        if sig == signal.SIGTERM:
            time.sleep(3)
    return [p['pid'] for p in victims]


def probe(directory):
    """Probe quota as well as free blocks; df alone cannot detect EDQUOT."""
    target = Path(directory) / f'.recovery_probe_{os.getpid()}'
    result = {'free_bytes': shutil.disk_usage(directory).free, 'writable': False}
    try:
        with target.open('wb') as stream:
            stream.write(b'quota-probe\n')
            stream.flush()
            os.fsync(stream.fileno())
        result['writable'] = True
    except OSError as exc:
        result['errno'] = exc.errno
    finally:
        target.unlink(missing_ok=True)
    return result


def release_reserve(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError('Unexpected recovery reserve')
    size = path.stat().st_size
    path.unlink()
    return size


def copy_closed(source, destination, expected, receipt):
    """Preserve exact data and its original path while freeing quota space."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        return 0
    before = source.stat()
    if before.st_uid != os.getuid() or before.st_size != expected['size'] or digest(source) != expected['sha256']:
        raise ValueError('Completed result differs from its audit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != expected['sha256']:
            raise ValueError('Conflicting recovery copy')
    else:
        temporary = destination.with_name(destination.name + '.copying')
        shutil.copy2(source, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        if digest(temporary) != expected['sha256']:
            raise ValueError('Recovery copy failed checksum validation')
        temporary.replace(destination)
    write(receipt, {'source': str(source), 'destination': str(destination), **expected,
                    'state': 'copied_and_verified', 'at': now()})
    current = source.stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError('Result changed during recovery')
    link = source.with_name(source.name + '.recovery_link')
    if link.is_symlink() and link.readlink() == destination:
        link.unlink()
    link.symlink_to(destination)
    link.replace(source)
    write(receipt, {'source': str(source), 'destination': str(destination), **expected,
                    'state': 'offloaded', 'at': now()})
    return before.st_size


def offload_completed_results(plan, limit=1024**3):
    root = Path(plan['dependency']['root'])
    if digest(root / 'plan.json') != plan['dependency']['plan_sha256']:
        raise ValueError('Evaluation plan changed')
    previous = read(root / 'plan.json')
    moved, count = 0, 0
    local_device = Path(plan['scratch']).stat().st_dev
    for stage in previous['stages']:
        stage_root = Path(stage['root'])
        if not stage_root.is_relative_to(root / 'stages'):
            raise ValueError('Invalid evaluation stage')
        manifest = read(stage_root / 'execution_manifest.json')
        for summary_path in sorted(stage_root.glob('results/*/*/budget_*/summary.json')):
            summary = read(summary_path)
            if (summary.get('state') != 'complete' or summary.get('ledger_audit') != 'passed'
                    or summary.get('identity', {}).get('manifest') != manifest['fingerprint']):
                continue
            for item in summary['artifacts'].values():
                relative = Path(item['file'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid completed result path')
                source = summary_path.parent / relative
                if (source.is_symlink() or source.stat().st_dev == local_device
                        or time.time() - source.stat().st_mtime < 120):
                    continue
                key = hashlib.sha256(str(source).encode()).hexdigest()
                base = Path(plan['scratch']) / 'recovered_results' / key
                moved += copy_closed(source, base / source.name, item, base / 'receipt.json')
                count += 1
                if moved >= limit or count >= 128:
                    return {'bytes': moved, 'files': count}
    # Failed attempts may lack a completed-point audit. Their gzip ledger is
    # written by one open stream; after that stream closes it is never appended
    # again (retries use a fresh UUID). Preserve those exact bytes too, rather
    # than letting old partial ledgers consume the remaining home quota.
    opened = set()
    for proc in processes():
        try:
            for fd in Path('/proc', str(proc['pid']), 'fd').iterdir():
                try:
                    opened.add(os.readlink(fd))
                except OSError:
                    pass
        except (PermissionError, FileNotFoundError):
            pass
    for stage in previous['stages']:
        for source in sorted(Path(stage['root']).glob('results/*/*/budget_*/attempt_*/rollouts.jsonl.gz')):
            if (source.is_symlink() or source.stat().st_dev == local_device or str(source) in opened
                    or str(source.resolve()) in opened or time.time() - source.stat().st_mtime < 300):
                continue
            expected = {'size': source.stat().st_size, 'sha256': digest(source), 'kind': 'closed_attempt_ledger'}
            base = Path(plan['scratch']) / 'recovered_results' / hashlib.sha256(str(source).encode()).hexdigest()
            moved += copy_closed(source, base / source.name, expected, base / 'receipt.json')
            count += 1
            if moved >= limit or count >= 128:
                return {'bytes': moved, 'files': count}
    return {'bytes': moved, 'files': count}


def recover_budget_controller(plan, snapshot):
    from qwen3_experiments.taco_grpo_pipeline import environment

    root = Path(plan['dependency']['root'])
    if digest(root / 'plan.json') != plan['dependency']['plan_sha256']:
        raise ValueError('Evaluation plan changed')
    current = read(root / 'queue_status.json')
    if current.get('state') == 'complete':
        return {'state': 'complete'}
    policy = plan['recovery_policy']
    parent = Path(policy['budget_supervisor'])
    if digest(parent) != policy['budget_supervisor_sha256']:
        raise ValueError('Evaluation supervisor changed')
    live = processes()
    parents = [p for p in live if str(parent) in p['args']]
    control = process(current.get('pid', 0))
    if control and control['uid'] == os.getuid():
        allowed = (('qwen3_experiments.taco_priority_queue' in control['args'] and str(parent.parent) in control['args'])
                   or ('qwen3_experiments.compression_budget_followup' in control['args'] and str(root) in control['args']
                       and 'queue' in control['args']))
        if not allowed:
            raise ValueError('Evaluation controller identity changed')
        stamp = datetime.fromisoformat(current['updated_at']).timestamp()
        if time.time() - stamp > 600 and snapshot['disks']['home']['writable']:
            # Its persistent parent restarts it; existing evaluation workers are
            # adopted by the queue through their frozen run identities.
            if process(control['pid']) == control:
                os.kill(control['pid'], signal.SIGTERM)
            return {'state': 'restarting_stale_controller', 'pid': control['pid']}
        return {'state': 'running', 'pid': control['pid']}
    if parents:
        return {'state': 'parent_restarting_controller', 'parent_pid': parents[0]['pid']}
    logfile = local_log(plan, 'recovery/budget_supervisor.log')
    with logfile.open('ab', buffering=0) as stream:
        child = subprocess.Popen(['bash', str(parent)], env=environment(plan), stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return {'state': 'restarted_supervisor', 'pid': child.pid}


def tick(plan):
    from qwen3_experiments.taco_grpo_pipeline import persist

    policy = plan['recovery_policy']
    snapshot = {'state': 'monitoring', 'pid': os.getpid(), 'updated_at': now(), 'actions': [], 'issues': []}
    snapshot['disks'] = {name: probe(path) for name, path in policy['write_probe_directories'].items()}
    pressure = any(not d['writable'] for d in snapshot['disks'].values())
    pressure |= snapshot['disks']['project']['free_bytes'] < policy['project_critical_bytes']
    pressure |= snapshot['disks']['compute']['free_bytes'] < policy['compute_critical_bytes']
    snapshot['disk_pressure'] = bool(pressure)
    if pressure:
        for reserve in policy['reserves']:
            try:
                reclaimed = release_reserve(reserve)
                if reclaimed:
                    snapshot['actions'].append({'action': 'release_reserve', 'path': reserve, 'bytes': reclaimed})
            except Exception as exc:
                snapshot['issues'].append(str(exc))
        if snapshot['disks']['compute']['free_bytes'] > policy['compute_critical_bytes']:
            try:
                result = offload_completed_results(plan)
                if result['bytes']:
                    snapshot['actions'].append({'action': 'offload_verified_results', **result})
            except Exception as exc:
                snapshot['issues'].append(str(exc))
    try:
        snapshot['budget_controller'] = recover_budget_controller(plan, snapshot)
    except Exception as exc:
        snapshot['issues'].append(str(exc))
    persist(plan, 'resilience_status.json', snapshot)
    return snapshot


def serve(plan):
    """Parent stays alive through full status/log filesystems and queue exits."""
    from qwen3_experiments.taco_grpo_pipeline import active, environment, persist, state, verify, MODULE

    if verify(Path(plan['output_root'])) != plan:
        raise ValueError('Supervisor plan differs from the frozen compute run')
    local = Path(plan['scratch']) / 'guard'
    local.mkdir(parents=True, exist_ok=True)
    children = []
    with (local / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                children = [child for child in children if child.poll() is None]
                tick(plan)
                if state(plan, 'queue_status.json').get('state') == 'complete':
                    persist(plan, 'supervisor_status.json', {'state': 'complete', 'pid': os.getpid()})
                    return
                queue = active(plan, 'queue')
                if not queue:
                    logfile = local_log(plan, 'recovery/queue.log')
                    with logfile.open('ab', buffering=0) as log:
                        child = subprocess.Popen([plan['python_bin'], '-u', '-m', MODULE, 'queue', '--root', plan['output_root']],
                                                 cwd=plan['runtime'], env=environment(plan), stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append(child)
                    queue = [child.pid]
                persist(plan, 'supervisor_status.json', {'state': 'supervising', 'pid': os.getpid(), 'queue_pids': queue})
            except Exception as exc:
                persist(plan, 'supervisor_status.json', {'state': 'recovering', 'pid': os.getpid(), 'error': str(exc)})
            time.sleep(plan['recovery_policy']['interval_seconds'])
