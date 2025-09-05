# backend/moderation_ml.py
from __future__ import annotations
from typing import Dict, Tuple
import re

# Non-LLM model: RoBERTa fine-tuned on Jigsaw (Detoxify)
from detoxify import Detoxify


SAFE_MESSAGES: Dict[str, str] = {
    "self_harm": (
        "I’m really sorry you’re feeling this way, but I can’t help with that. "
        "If you’re in immediate danger or considering self-harm, please contact your local emergency number "
        "or speak to a qualified professional right now."
    ),
    "sexual_minors": "I can’t assist with sexual content involving minors.",
    "violence_illegal": "I can’t assist with requests that could harm you or others or are illegal.",
    "medical_harm": "I can’t assist with dangerous medical instructions or overdose guidance.",
    "toxicity": "I’m not able to continue with abusive or threatening language.",
    "nsfw": "I can’t assist with sexually explicit content.",
    "general": "Sorry, I can’t help with this request.",
}


class ModerationEnsemble:
    """
    Non-LLM moderation gate: Detoxify (RoBERTa on Jigsaw) + high-precision rules → allow/block.
    Runs BEFORE any retrieval or PubMed calls.
    """

    def __init__(
        self,
        detoxify_variant: str = "original",  # 'original' covers toxicity/threat/identity/insult/obscene/sexual_explicit
        thresholds: Dict[str, float] | None = None,
    ):
        # Load classifier
        self.detox = Detoxify(detoxify_variant)

        # Thresholds tuned for strong recall on harmful content; you can adjust later.
        self.t = thresholds or {
            "toxicity": 0.60,
            "severe_toxicity": 0.40,
            "threat": 0.35,
            "identity_attack": 0.50,
            "insult": 0.65,
            "obscene": 0.65,
            "sexual_explicit": 0.40,
        }

        # -------- High-precision RULES (hard blocks) --------
        # Self-harm
        self.re_self_harm = re.compile(
            r"(kill myself|suicide|end my life|self[-\s]?harm|hurt myself|cutting|self[-\s]?mutilation)",
            re.IGNORECASE,
        )
        # Medical harm / overdose specifics
        self.re_med_harm = re.compile(
            r"(overdose|lethal dose|how many (?:pills|tablets) (?:to|would) (?:kill|harm)|"
            r"(?:is it safe to )?mix .* (?:to overdose))",
            re.IGNORECASE,
        )
        # Violence / illegal weapons / explosives
        self.re_violence_illegal = re.compile(
            r"(kill (?:someone|people)|how to (?:make|build) (?:a )?(?:bomb|explosive|molotov|napalm)|"
            r"buy (?:unregistered|illegal) (?:gun|weapon)|how to (?:hack|scam|forge|cook meth))",
            re.IGNORECASE,
        )
        # Sexual content involving minors (age cue + sexual cue must co-occur)
        self.re_age_minor = re.compile(
            r"(\b(?:minor|underage|child|children|12|13|14|15|16|17)\b|(\d{1,2})\s?(?:yo|yrs?|years? old))",
            re.IGNORECASE,
        )
        self.re_sexual = re.compile(r"(sex|nude|naked|porn|explicit|xxx|erotic)", re.IGNORECASE)

    def decide(self, text: str) -> Tuple[bool, str, str, Dict]:
        """
        Decide if text should be blocked.

        Returns:
            blocked (bool), category (str), safe_message (str), details (dict with model scores & rule hits)
        """
        text = (text or "").strip()
        if not text:
            return False, "allow", "", {"reason": "empty"}

        # 1) Model probabilities
        det_scores = self._detox_scores(text)

        # 2) High-precision rule hits
        rules = self._rule_hits(text)

        # 3) Decision (severity order)
        if rules["sexual_minors"]:
            return True, "sexual_minors", SAFE_MESSAGES["sexual_minors"], self._pack(det_scores, rules)
        if rules["self_harm"]:
            return True, "self_harm", SAFE_MESSAGES["self_harm"], self._pack(det_scores, rules)
        if rules["violence_illegal"]:
            return True, "violence_illegal", SAFE_MESSAGES["violence_illegal"], self._pack(det_scores, rules)
        if rules["medical_harm"]:
            return True, "medical_harm", SAFE_MESSAGES["medical_harm"], self._pack(det_scores, rules)

        if det_scores.get("sexual_explicit", 0.0) >= self.t["sexual_explicit"]:
            return True, "nsfw", SAFE_MESSAGES["nsfw"], self._pack(det_scores, rules)
        if det_scores.get("threat", 0.0) >= self.t["threat"]:
            return True, "violence_illegal", SAFE_MESSAGES["violence_illegal"], self._pack(det_scores, rules)
        if det_scores.get("severe_toxicity", 0.0) >= self.t["severe_toxicity"]:
            return True, "toxicity", SAFE_MESSAGES["toxicity"], self._pack(det_scores, rules)

        if (
            det_scores.get("toxicity", 0.0) >= self.t["toxicity"]
            or det_scores.get("obscene", 0.0) >= self.t["obscene"]
            or det_scores.get("insult", 0.0) >= self.t["insult"]
            or det_scores.get("identity_attack", 0.0) >= self.t["identity_attack"]
        ):
            return True, "toxicity", SAFE_MESSAGES["toxicity"], self._pack(det_scores, rules)

        # Allow by default
        return False, "allow", "", self._pack(det_scores, rules)

    # ----- internals -----
    def _detox_scores(self, text: str) -> Dict[str, float]:
        # Detoxify returns a dict of probabilities
        out = self.detox.predict(text)
        return {k: float(v) for k, v in out.items()}

    def _rule_hits(self, text: str) -> Dict[str, bool]:
        sexual_minors = bool(self.re_age_minor.search(text) and self.re_sexual.search(text))
        return {
            "self_harm": bool(self.re_self_harm.search(text)),
            "medical_harm": bool(self.re_med_harm.search(text)),
            "violence_illegal": bool(self.re_violence_illegal.search(text)),
            "sexual_minors": sexual_minors,
        }

    def _pack(self, det_scores: Dict[str, float], rules: Dict[str, bool]) -> Dict:
        return {"detoxify": det_scores, "rules": rules}
