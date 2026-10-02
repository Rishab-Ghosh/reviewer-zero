import threading
import time

import reviewer.novelty.rerank as rr
from reviewer.models import TokenUsage


def test_judge_all_batches_overlap_and_keep_batch_order(monkeypatch):
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def fake_batch(client, query, batch, *, prompt, model, max_retries):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03 * (5 - batch[0] // 20))  # later batches finish first
        with lock:
            state["now"] -= 1
        return [("judged", c) for c in batch], TokenUsage(input_tokens=len(batch))

    monkeypatch.setattr(rr, "judge_batch", fake_batch)
    pool = list(range(95))
    seq, seq_usage = rr.judge_all(
        None, None, pool, prompt=None, model="m", batch_size=20, max_retries=0, workers=1
    )
    state["peak"] = 0
    par, par_usage = rr.judge_all(
        None, None, pool, prompt=None, model="m", batch_size=20, max_retries=0, workers=8
    )
    assert state["peak"] == 5  # all five batches in flight together
    assert par == seq == [("judged", c) for c in pool]
    assert par_usage.input_tokens == seq_usage.input_tokens == 95
