"""
Agentic care-plan generator for Dr. Charlotte.

Uses an OpenAI tool-calling loop to gather NHS/NICE guidelines and PubMed
evidence before synthesising a structured, evidence-based care plan.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import openai

from backend.official_guidance import OfficialGuidanceEngine
from backend.pubmed_search import PubMedCentralSearcher

_TOOLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_nhs_guidelines",
            "description": (
                "Search NHS.uk and NICE for evidence-based clinical guidelines, monitoring "
                "targets, treatment thresholds, and care pathways for a given condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "Condition name, e.g. 'Type 2 Diabetes' or 'Hypertension'"
                    },
                    "aspect": {
                        "type": "string",
                        "description": (
                            "Specific aspect to retrieve, e.g. 'monitoring targets', "
                            "'medication guidance', 'when to refer', 'lifestyle advice', "
                            "'escalation criteria'"
                        )
                    }
                },
                "required": ["condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed_evidence",
            "description": (
                "Search PubMed Central for systematic reviews, RCTs, and meta-analyses "
                "on clinical interventions, lifestyle modifications, or monitoring strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query"
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "The evidence focus, e.g. 'diet interventions', 'exercise RCT', "
                            "'sleep quality', 'medication adherence', 'self-monitoring outcomes'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_lifestyle_recommendations",
            "description": (
                "Retrieve evidence-based lifestyle guidance for a specific domain "
                "(diet, exercise, sleep, weight, stress, alcohol, smoking) for a condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "lifestyle_area": {
                        "type": "string",
                        "description": (
                            "One of: 'diet and nutrition', 'physical activity', "
                            "'sleep hygiene', 'weight management', 'stress and mental health', "
                            "'alcohol', 'smoking cessation'"
                        )
                    }
                },
                "required": ["condition", "lifestyle_area"]
            }
        }
    },
]

_PLAN_SCHEMA = {
    "type": "object",
    "required": [
        "condition", "title", "goals", "daily_tasks", "weekly_tasks",
        "medication_reminders", "lab_reminders", "escalation_thresholds",
        "lifestyle", "missed_care_checklist", "evidence_summary", "safety_notes"
    ],
    "additionalProperties": True,
    "properties": {
        "condition": {"type": "string"},
        "title": {"type": "string"},
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "metric": {"type": "string"},
                    "target_months": {"type": "integer"}
                }
            }
        },
        "daily_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time_of_day": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "bedtime", "any"]
                    },
                    "rationale": {"type": "string"}
                }
            }
        },
        "weekly_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rationale": {"type": "string"}
                }
            }
        },
        "medication_reminders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "dose": {"type": "string"},
                    "timing": {"type": "string"},
                    "notes": {"type": "string"}
                }
            }
        },
        "lab_reminders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test": {"type": "string"},
                    "frequency_months": {"type": "integer"},
                    "notes": {"type": "string"},
                    "target_value": {"type": "string"}
                }
            }
        },
        "escalation_thresholds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string"},
                    "threshold": {"type": "string"},
                    "action": {"type": "string"},
                    "urgency": {
                        "type": "string",
                        "enum": ["call_999", "a_and_e", "gp_same_day", "gp_routine", "self_monitor"]
                    }
                }
            }
        },
        "lifestyle": {
            "type": "object",
            "properties": {
                "diet": {"type": "string"},
                "exercise": {"type": "string"},
                "sleep": {"type": "string"},
                "weight": {"type": "string"},
                "mental_health": {"type": "string"},
                "smoking": {"type": "string"},
                "alcohol": {"type": "string"},
                "other": {"type": "string"}
            }
        },
        "missed_care_checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "frequency_months": {"type": "integer"},
                    "notes": {"type": "string"}
                }
            }
        },
        "evidence_summary": {"type": "string"},
        "safety_notes": {"type": "string"}
    }
}


class CarePlanAgent:
    MAX_ITERATIONS = 7

    def __init__(self) -> None:
        self._client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._guidance = OfficialGuidanceEngine()
        self._pubmed = PubMedCentralSearcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        condition: str,
        user_context: Dict,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        def emit(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        profile = user_context.get("profile", {})
        meds = user_context.get("medications", [])
        existing_conditions = user_context.get("conditions", [])
        chat_summary = (user_context.get("chat_summary") or "")[:800]
        dob = profile.get("date_of_birth") or "not recorded"
        sex = profile.get("biological_sex") or "not recorded"
        role = profile.get("role") or "Patient"

        system_prompt = f"""You are a specialist evidence-based care-plan assistant embedded in Dr. Charlotte, a UK-focused clinical AI platform (NHS-aligned).

Build a comprehensive, personalised care plan for: **{condition}**

Patient context:
- Date of birth: {dob}
- Biological sex: {sex}
- Account role: {role}
- Current medications: {json.dumps(meds) if meds else "none recorded"}
- Other known conditions: {json.dumps(existing_conditions) if existing_conditions else "none recorded"}
- Recent health chat context: {chat_summary if chat_summary else "none"}

AGENT RULES:
1. Use tools to gather real evidence — do not invent guidelines or statistics.
2. Search at minimum: (a) NHS/NICE monitoring targets and treatment thresholds, (b) lifestyle evidence (diet, exercise, sleep), (c) escalation/red-flag criteria.
3. For each lifestyle domain (diet, exercise, sleep, weight, mental health, smoking, alcohol) provide specific, actionable, evidence-grounded advice — minimum 2-3 sentences per domain.
4. Daily tasks must be concrete and achievable (e.g. "Take metformin 500mg with breakfast" — not "manage diabetes").
5. Lab reminders must use correct NHS/NICE frequencies in months.
6. Escalation thresholds must state real clinical values (e.g. "BP > 180/120 mmHg" not "very high BP").
7. Safety notes must flag medication interactions, red flags, and safeguarding concerns specific to this condition.
8. After gathering evidence, generate the final plan JSON only — no prose outside the JSON."""

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a full, evidence-based care plan for: {condition}"},
        ]

        emit(f"Starting evidence search for {condition}...")

        for _ in range(self.MAX_ITERATIONS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0.15,
            )
            msg = response.choices[0].message

            # Build the assistant message dict manually (avoids Pydantic serialisation issues)
            assistant_msg: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fname = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if fname == "search_nhs_guidelines":
                    cond = args.get("condition", condition)
                    aspect = args.get("aspect", "")
                    emit(f"Searching NHS/NICE: {cond} — {aspect or 'guidelines'}")
                    result = self._nhs(f"{cond} {aspect}".strip())

                elif fname == "search_pubmed_evidence":
                    query = args.get("query", condition)
                    emit(f"Searching PubMed: {query}")
                    result = self._pubmed_search(query)

                elif fname == "search_lifestyle_recommendations":
                    cond = args.get("condition", condition)
                    area = args.get("lifestyle_area", "lifestyle")
                    emit(f"Searching lifestyle evidence: {area} for {cond}")
                    result = self._nhs(f"{cond} {area} recommendations")

                else:
                    result = "Unknown tool."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result or "No relevant results found.",
                })

        emit("Synthesising your personalised care plan...")

        messages.append({
            "role": "user",
            "content": (
                "Now produce the final care plan as a valid JSON object matching this schema:\n"
                f"{json.dumps(_PLAN_SCHEMA, indent=2)}\n\n"
                "Requirements:\n"
                "- goals: 3-5 specific measurable goals with target_months\n"
                "- daily_tasks: 4-8 actionable daily actions with time_of_day set\n"
                "- weekly_tasks: 3-5 actions\n"
                "- lab_reminders: ALL NHS/NICE recommended tests with correct frequency_months and target_value\n"
                "- escalation_thresholds: 4-6 items with real clinical values in threshold field\n"
                "- lifestyle: populate ALL 7 fields (diet, exercise, sleep, weight, mental_health, smoking, alcohol) with at least 2-3 sentences each\n"
                "- missed_care_checklist: include annual review, vaccinations, and condition-specific screenings\n"
                "- evidence_summary: cite NICE guideline number (e.g. NG28) and year if known\n"
                "Return ONLY the JSON object."
            ),
        })

        final = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.05,
        )

        raw = final.choices[0].message.content or "{}"
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {}

        now = datetime.now(timezone.utc).isoformat()
        plan.setdefault("condition", condition)
        plan.setdefault("title", f"{condition} Care Plan")
        plan.setdefault("goals", [])
        plan.setdefault("daily_tasks", [])
        plan.setdefault("weekly_tasks", [])
        plan.setdefault("medication_reminders", [])
        plan.setdefault("lab_reminders", [])
        plan.setdefault("escalation_thresholds", [])
        plan.setdefault("lifestyle", {})
        plan.setdefault("missed_care_checklist", [])
        plan.setdefault("evidence_summary", "Generated from NHS/NICE guidelines and PubMed evidence.")
        plan.setdefault("safety_notes", "Always consult your GP or healthcare professional before making changes to your care.")
        plan.update({
            "id": uuid.uuid4().hex,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "after_visit_notes": [],
            "gp_prep_summary": None,
        })

        for key in ("goals", "daily_tasks", "weekly_tasks", "medication_reminders",
                    "lab_reminders", "escalation_thresholds", "missed_care_checklist"):
            for item in plan.get(key, []):
                item.setdefault("id", uuid.uuid4().hex[:12])

        return plan

    def generate_gp_prep(self, plan: Dict, user_context: Dict) -> str:
        profile = user_context.get("profile", {})
        name = (profile.get("display_name") or "the patient").split()[0]

        prompt = (
            f"You are Dr. Charlotte helping {name} prepare for their GP appointment about "
            f"{plan.get('condition', 'their condition')}.\n\n"
            f"Care plan summary:\n"
            f"Goals: {json.dumps(plan.get('goals', []))}\n"
            f"Medications: {json.dumps(plan.get('medication_reminders', []))}\n"
            f"Upcoming labs: {json.dumps(plan.get('lab_reminders', []))}\n"
            f"Warning signs: {json.dumps(plan.get('escalation_thresholds', []))}\n"
            f"Missed care: {json.dumps(plan.get('missed_care_checklist', []))}\n"
            f"After-visit notes: {json.dumps(plan.get('after_visit_notes', []))}\n\n"
            "Write a structured GP appointment preparation guide with these sections:\n"
            "## Questions to ask your GP (5-7 specific questions)\n"
            "## Medications to review\n"
            "## Tests and results to request\n"
            "## Symptoms or concerns to mention\n"
            "## What you want to achieve from this appointment\n\n"
            "Keep it concise, patient-friendly, and in plain language."
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _nhs(self, query: str) -> str:
        try:
            results = self._guidance.search([query], per_source_limit=2)
            if not results:
                return "No NHS/NICE results found."
            parts = []
            for r in results[:4]:
                title = r.get("title", "")
                snippet = (r.get("snippet") or r.get("content") or "")[:500]
                url = r.get("url", "")
                parts.append(f"[{title}]\n{snippet}\nSource: {url}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"NHS search error: {exc}"

    def _pubmed_search(self, query: str) -> str:
        try:
            records = self._pubmed.search_article_records(query, 3)
            if not records:
                return "No PubMed results found."
            parts = []
            for r in records[:3]:
                title = r.get("title", "")
                abstract = (r.get("abstract") or "")[:450]
                year = r.get("year", "")
                journal = r.get("journal", "")
                parts.append(f"[{title} — {journal} {year}]\n{abstract}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"PubMed error: {exc}"
