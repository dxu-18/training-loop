"""torch-lite profiler: counts events (record_function regions and a few aten-style ops).
Times are real wall-clock measurements, so only names and counts are reproducible."""

import contextlib
import enum
import json
import time

import torch


class ProfilerActivity:
    CPU, CUDA = "CPU", "CUDA"


class ProfilerAction(enum.Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def schedule(*, wait, warmup, active, repeat=0, skip_first=0):
    def fn(step):
        if step < skip_first:
            return ProfilerAction.NONE
        step -= skip_first
        cycle = wait + warmup + active
        if repeat and step >= repeat * cycle:
            return ProfilerAction.NONE
        pos = step % cycle
        if pos < wait:
            return ProfilerAction.NONE
        if pos < wait + warmup:
            return ProfilerAction.WARMUP
        return ProfilerAction.RECORD_AND_SAVE if pos == cycle - 1 else ProfilerAction.RECORD
    return fn


class _Avg:
    def __init__(self, key):
        self.key, self.count, self.cpu_time_total, self.self_cpu_time_total = key, 0, 0.0, 0.0

    @property
    def cpu_time(self):
        return self.cpu_time_total / max(self.count, 1)

    def __repr__(self):
        return f"<FunctionEventAvg key={self.key} count={self.count}>"


class EventList(list):
    def table(self, sort_by=None, row_limit=100, **kw):
        rows = sorted(self, key=lambda e: -getattr(e, sort_by.replace("cuda", "cpu"), 0)) if sort_by else list(self)
        lines = [f"{'Name':<32}{'# of Calls':>12}{'CPU total (us)':>18}", "-" * 62]
        lines += [f"{e.key:<32}{e.count:>12}{e.cpu_time_total:>18.1f}" for e in rows[:row_limit]]
        return "\n".join(lines)


class profile:
    def __init__(self, activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False,
                 with_stack=False, with_flops=False):
        self.activities = activities or [ProfilerActivity.CPU]
        self._schedule, self._on_ready = schedule, on_trace_ready
        self.step_num = 0
        self._events = {}
        self._open = []

    def _recording(self):
        return self._schedule is None or self._schedule(self.step_num) in (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)

    def _record(self, name, seconds=0.0):
        if self._recording():
            ev = self._events.setdefault(name, _Avg(name))
            ev.count += 1
            ev.cpu_time_total += seconds * 1e6
            ev.self_cpu_time_total += seconds * 1e6

    def __enter__(self):
        torch._PROF.append(self)
        return self

    def __exit__(self, *exc):
        if self in torch._PROF:
            torch._PROF.remove(self)
        if self._on_ready and self._schedule is None:
            self._on_ready(self)
        return False

    def start(self):
        return self.__enter__()

    def stop(self):
        return self.__exit__(None, None, None)

    def step(self):
        if self._schedule and self._schedule(self.step_num) == ProfilerAction.RECORD_AND_SAVE and self._on_ready:
            self._on_ready(self)
        self.step_num += 1

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        return EventList(self._events.values())

    def events(self):
        return self.key_averages()

    def export_chrome_trace(self, path):
        with open(path, "w") as fh:
            json.dump({"traceEvents": [{"name": e.key, "ph": "X", "args": {"count": e.count}} for e in self._events.values()]}, fh)


class record_function(contextlib.ContextDecorator):
    def __init__(self, name, args=None):
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0
        for p in torch._PROF:
            p._record(self.name, dt)
        return False
