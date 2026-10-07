"""Tiny CPU-only MLX compatibility checks; never load weights or private images."""
from __future__ import annotations

import argparse
import json
import sys

from repo_layout import REPO_ROOT, SOURCE_ROOT

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))


def check(model_path: str) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    from PIL import Image
    from transformers import AutoProcessor

    from specialization_training import restore_rng
    from specialization_worker import configure_pixels

    with mx.stream(mx.cpu):
        mx.random.seed(17)
        keys = [mx.array(k) for k in mx.random.state]
        first = mx.random.uniform(shape=(4,))
        mx.eval(first)
        restore_rng(mx, keys)
        replay = mx.random.uniform(shape=(4,))
        mx.eval(replay)
        assert bool(mx.all(first == replay).item())
        logits = mx.array([[[.1, .2, .3], [.4, .5, .2], [.3, .6, .1], [.2, .1, .7]]])
        labels = mx.array([[0, 1, 2, 0]], dtype=mx.int32)
        mask = mx.array([[0, 0, 1, 1]], dtype=mx.float32)
        indices = mx.array([2, 3], dtype=mx.int32)
        def full(value):
            return (nn.losses.cross_entropy(value.astype(mx.float32), labels)*mask).sum()/mask.sum()
        def selected(value):
            return nn.losses.cross_entropy(mx.take(value, indices, axis=1).astype(mx.float32),
                                           mx.take(labels, indices, axis=1)).mean()
        a, b = full(logits), selected(logits)
        ga, gb = mx.grad(full)(logits), mx.grad(selected)(logits)
        mx.eval(a, b, ga, gb)
        assert float(abs(a-b).item()) == 0
        assert float(mx.max(mx.abs(ga-gb)).item()) == 0
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        counts = {}
        for budget in (512, 1024, 1536):
            configure_pixels(processor, budget)
            prepared = processor.image_processor(images=[Image.new('RGB', (1600, 1000), 'white')],
                                                  return_tensors='np')
            tokens = int(prepared['image_grid_thw'][0].prod()) // processor.image_processor.merge_size**2
            assert 0 < tokens <= budget
            counts[str(budget)] = tokens
    return {'status': 'passed', 'tiny_cpu_only': True, 'loads_model': False,
            'reads_private_images': False, 'rng_replay': True, 'assistant_loss_gradient_parity': True,
            'visual_tokens_by_budget': counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.model_path)))
