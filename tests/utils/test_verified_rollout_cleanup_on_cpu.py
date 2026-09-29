import hashlib
import json
from types import SimpleNamespace

import pytest

from qwen3_experiments.verified_rollout_cleanup import cleanup


@pytest.mark.parametrize('valid', [False, True])
def test_keep_shard_until_hub_hash_matches(tmp_path, valid):
    data = tmp_path / 'rollout' / 'data'
    data.mkdir(parents=True)
    shard = data / 'step_000001.jsonl.gz'
    shard.write_bytes(b'closed rollout')
    manifest = {'num_steps': 1, 'num_rollouts': 1, 'steps': {'1': {'file': 'data/step_000001.jsonl.gz'}}}
    (data.parent / 'rollout_manifest.json').write_text(json.dumps(manifest))
    cfg = {'rollout_directory': str(data.parent), 'steps': 1, 'rollouts': 1,
           'repo_id': 'test/rollouts', 'allowed_data_roots': [str(data)],
           'receipt_directories': [str(tmp_path / 'receipts')]}
    digest = hashlib.sha256(shard.read_bytes()).hexdigest() if valid else '0' * 64
    item = SimpleNamespace(rfilename='data/step_000001.jsonl.gz', size=shard.stat().st_size,
                           lfs=SimpleNamespace(sha256=digest))
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=False, sha='a' * 40, siblings=[item]))
    if valid:
        result = cleanup(cfg, api)
        assert result['state'] == 'uploaded_verified_and_deleted'
        assert not shard.exists()
    else:
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            cleanup(cfg, api)
        assert shard.exists()
