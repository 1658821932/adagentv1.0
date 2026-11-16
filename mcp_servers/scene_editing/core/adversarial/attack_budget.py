"""Cumulative attack budgets shared by retries in one main-Agent execution scope."""
import fcntl
import json
import os
import time
import contextvars
from pathlib import Path
from contextlib import contextmanager
from experiment_runtime import atomic_json


class AttackBudgetExhausted(RuntimeError):
    pass


AUTO_SESSION_ROOT=contextvars.ContextVar('attack_auto_session_root',default=None)


def constructor_budget_path(run_dir):
    root = os.getenv('ADVERSARIAL_ATTACK_SESSION') or AUTO_SESSION_ROOT.get()
    return Path(root)/'constructor-budget.json' if root else Path(run_dir)/'constructor-budget.json'


class BudgetedArchive:
    """Keep request count/time across fresh backend processes; no credentials stored."""
    def __init__(self, archive, output, max_requests=18, max_api_seconds=900):
        self.archive=archive
        self.path=Path(os.getenv('ADVERSARIAL_ATTACK_SESSION') or AUTO_SESSION_ROOT.get() or str(output))/'api-budget.json'
        self.max_requests=max_requests
        self.max_api_seconds=max_api_seconds
        self.closing=False

    def __getattr__(self,name):
        return getattr(self.archive,name)

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            state=json.loads(self.path.read_text()) if self.path.exists() else dict(requests=0,api_seconds=0.)
            yield state
            atomic_json(self.path,state)

    def context(self):
        with self.locked() as state:
            return dict(requests=state['requests'],api_seconds=round(state['api_seconds'],3),
                requests_remaining=max(0,self.max_requests-state['requests']),
                api_seconds_remaining=max(0,round(self.max_api_seconds-state['api_seconds'],3)),
                max_requests=self.max_requests,max_api_seconds=self.max_api_seconds,
                finalization_reserved_requests=2,finalization_reserved_api_seconds=90,
                search_requests_remaining=max(0,self.max_requests-state['requests']-2),
                search_api_seconds_remaining=max(0,round(self.max_api_seconds-state['api_seconds']-90,3)),
                scope='cumulative across attack retries in the same main-Agent execution scope')

    def complete(self,label,*args,**kwargs):
        with self.locked() as state:
            # A killed request is charged once, bounded by its recorded timeout.
            active=state.pop('active_request',None)
            if active:
                state['api_seconds']+=min(active['timeout'],max(0,time.time()-active['started_at']))
            remaining=self.max_api_seconds-state['api_seconds']-(0 if self.closing else 90)
            slots=self.max_requests-state['requests']-(0 if self.closing else 2)
            if remaining<=0 or slots<=0:
                # Persist recovery accounting even when the next request is refused.
                atomic_json(self.path,state)
                raise AttackBudgetExhausted('Cumulative attack search/API budget exhausted; any reserved allowance is only for final review of existing qualified candidates, not a fresh search')
            kwargs['total_seconds']=min(float(kwargs.get('total_seconds',remaining)),remaining)
            kwargs['no_output_seconds']=min(float(kwargs.get('no_output_seconds',60)),kwargs['total_seconds'])
            state['requests']+=1
            state['active_request']=dict(label=label,started_at=time.time(),timeout=kwargs['total_seconds'])
        started=time.perf_counter()
        try:
            return self.archive.complete(label,*args,**kwargs)
        finally:
            with self.locked() as state:
                state.pop('active_request',None)
                state['api_seconds']+=time.perf_counter()-started
