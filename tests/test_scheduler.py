import pytest

from mini_vllm.scheduler import Request, Scheduler


def test_add_request_goes_to_waiting():
    s = Scheduler()
    req = s.add_request(prompt_len=10)
    assert req.status == "WAITING"
    assert s.waiting == [req]
    assert not s.running


def test_schedule_admits_within_budget():
    s = Scheduler(block_size=4, max_prefill_tokens=100)
    r1 = s.add_request(prompt_len=5)
    r2 = s.add_request(prompt_len=3)
    new, running = s.schedule(free_blocks=100)
    assert new == [r1, r2]
    assert running == [r1, r2]
    assert r1.status == r2.status == "RUNNING"
    assert not s.waiting


def test_schedule_respects_prefill_budget():
    s = Scheduler(block_size=4, max_prefill_tokens=8)
    r1 = s.add_request(prompt_len=5)
    r2 = s.add_request(prompt_len=4)   # 5 + 4 = 9 > 8
    new, _ = s.schedule(free_blocks=100)
    assert new == [r1]
    assert s.waiting == [r2]


def test_schedule_respects_kv_block_budget():
    s = Scheduler(block_size=4)
    r1 = s.add_request(prompt_len=9)   # 3 blocks
    r2 = s.add_request(prompt_len=5)   # 2 blocks
    new, _ = s.schedule(free_blocks=4)  # only 4 blocks free
    assert new == [r1]
    assert s.waiting == [r2]


def test_schedule_does_not_resume_when_waiting_again():
    s = Scheduler(block_size=4)
    r1 = s.add_request(prompt_len=5)
    s.schedule(free_blocks=100)
    assert s.preempt() is r1
    assert r1.status == "WAITING"
    assert r1.kv_blocks == 0
    assert s.waiting[0] is r1
    assert not s.running


def test_preempt_picks_newest_running_request():
    s = Scheduler()
    r1 = s.add_request(prompt_len=2)
    r2 = s.add_request(prompt_len=2)
    s.schedule(free_blocks=100)
    assert s.preempt() is r2
    assert s.preempt() is r1
    assert s.preempt() is None


def test_finish_removes_from_running():
    s = Scheduler()
    r1 = s.add_request(prompt_len=3, max_new_tokens=1)
    s.schedule(free_blocks=100)
    s.finish(r1)
    assert r1.status == "FINISHED"
    assert s.running == []
    assert s.finished == [r1]
    assert not s.has_running_requests()


def test_generation_progress():
    s = Scheduler()
    req = s.add_request(prompt_len=4, max_new_tokens=3)
    s.schedule(free_blocks=100)
    assert req.num_generated == 0
    for i in range(1, 4):
        req.num_generated += 1
        assert req.num_generated == i
    assert req.num_generated == req.max_new_tokens


def test_preempted_request_can_be_rescheduled():
    s = Scheduler(block_size=4)
    r1 = s.add_request(prompt_len=8)   # 2 blocks
    s.schedule(free_blocks=4)
    s.preempt()
    # after blocks are freed, it can be admitted again
    new, _ = s.schedule(free_blocks=4)
    assert new == [r1]
    assert r1.status == "RUNNING"
