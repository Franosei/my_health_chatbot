from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_STORE_DIR = Path("data/care_plans")


def _ensure() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)


def _path(username: str) -> Path:
    return _STORE_DIR / f"{username}.json"


def _load(username: str) -> List[Dict]:
    _ensure()
    p = _path(username)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return []


def _save(username: str, plans: List[Dict]) -> None:
    _ensure()
    _path(username).write_text(json.dumps(plans, indent=2, ensure_ascii=False), "utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return date.today().isoformat()


def _stamp_ids(plan: Dict) -> Dict:
    for key in (
        "goals", "daily_tasks", "weekly_tasks", "medication_reminders",
        "lab_reminders", "escalation_thresholds", "missed_care_checklist",
    ):
        for item in plan.get(key, []):
            if not item.get("id"):
                item["id"] = uuid.uuid4().hex[:12]
    return plan


class CarePlanStore:
    @staticmethod
    def list_plans(username: str) -> List[Dict]:
        return _load(username)

    @staticmethod
    def get_plan(username: str, plan_id: str) -> Optional[Dict]:
        return next((p for p in _load(username) if p.get("id") == plan_id), None)

    @staticmethod
    def save_plan(username: str, plan: Dict) -> Dict:
        plans = _load(username)
        plan = _stamp_ids(plan)
        plan["updated_at"] = _now()
        idx = next((i for i, p in enumerate(plans) if p.get("id") == plan.get("id")), None)
        if idx is not None:
            plans[idx] = plan
        else:
            plans.append(plan)
        _save(username, plans)
        return plan

    @staticmethod
    def delete_plan(username: str, plan_id: str) -> bool:
        plans = _load(username)
        filtered = [p for p in plans if p.get("id") != plan_id]
        if len(filtered) == len(plans):
            return False
        _save(username, filtered)
        return True

    @staticmethod
    def toggle_task(username: str, plan_id: str, task_id: str, done: bool) -> Optional[Dict]:
        plan = CarePlanStore.get_plan(username, plan_id)
        if not plan:
            return None
        today = _today()
        for task in plan.get("daily_tasks", []) + plan.get("weekly_tasks", []):
            if task.get("id") == task_id:
                completed: List[str] = task.setdefault("completed_dates", [])
                if done and today not in completed:
                    completed.append(today)
                elif not done and today in completed:
                    completed.remove(today)
                break
        return CarePlanStore.save_plan(username, plan)

    @staticmethod
    def add_after_visit_note(username: str, plan_id: str, note: str) -> Optional[Dict]:
        plan = CarePlanStore.get_plan(username, plan_id)
        if not plan:
            return None
        plan.setdefault("after_visit_notes", []).append({"text": note, "date": _today()})
        return CarePlanStore.save_plan(username, plan)

    @staticmethod
    def set_gp_prep(username: str, plan_id: str, summary: str) -> Optional[Dict]:
        plan = CarePlanStore.get_plan(username, plan_id)
        if not plan:
            return None
        plan["gp_prep_summary"] = summary
        return CarePlanStore.save_plan(username, plan)
