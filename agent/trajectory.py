"""Research trajectory manager — persistent memory for the agent.

Stores every research direction, its attempts, backtest results,
and learnings. Feeds back into hypothesis generation so the agent
evolves its research trajectory instead of blindly tweaking code.
"""

import json
import os
from datetime import datetime


class ResearchTrajectory:
    def __init__(self, path: str = "trajectory.json"):
        self.path = path
        self.data = {"directions": [], "updated": None}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {"directions": [], "updated": None}

    def save(self):
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_direction(self, name: str) -> dict:
        for d in self.data["directions"]:
            if d["name"] == name:
                return d
        return None

    def add_direction(self, name: str, hypothesis: str):
        if self.get_direction(name) is None:
            self.data["directions"].append({
                "name": name,
                "hypothesis": hypothesis,
                "status": "active",
                "attempts": [],
            })
        self.save()

    def add_attempt(self, direction_name: str, factor_formula: str,
                    is_result: dict, oos_result: dict, learning: str):
        d = self.get_direction(direction_name)
        if d is None:
            d = {"name": direction_name, "hypothesis": "",
                 "status": "active", "attempts": []}
            self.data["directions"].append(d)
        d["attempts"].append({
            "formula": factor_formula,
            "is_summary": {
                "sharpe_max": is_result.get("sharpe_max"),
                "roughness": is_result.get("roughness", {}).get("combined"),
            },
            "oos_summary": oos_result,
            "learning": learning,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()

    def update_status(self, direction_name: str, status: str):
        d = self.get_direction(direction_name)
        if d:
            d["status"] = status
            self.save()

    def direction_context(self, name: str, max_attempts: int = 3) -> str:
        """
        Formatted history of recent attempts for ONE direction.

        Returns a string for injection into the Factor Generator's
        user message, or a placeholder if the direction is new.
        """
        d = self.get_direction(name)
        if not d or not d["attempts"]:
            return "（该方向此前未尝试过）"

        lines = []
        for a in d["attempts"][-max_attempts:]:
            is_s = a["is_summary"]
            oos_s = a["oos_summary"]
            lines.append(f"  - 公式: {a['formula'][:100]}")
            lines.append(f"    IS Sharpe max={is_s.get('sharpe_max')}, "
                         f"粗糙度={is_s.get('roughness')}, OOS={oos_s}")
            if a["learning"]:
                lines.append(f"    教训: {a['learning']}")
        return "\n".join(lines)

    def summary(self, max_directions: int = 12) -> str:
        """Compact summary for LLM consumption."""
        lines = []
        for d in self.data["directions"][:max_directions]:
            n_attempts = len(d["attempts"])
            lines.append(f"方向「{d['name']}」[{d['status']}] 尝试{n_attempts}次: {d['hypothesis']}")
            for a in d["attempts"][-2:]:
                is_s = a["is_summary"]
                oos_s = a["oos_summary"]
                lines.append(f"  - IS Sharpe={is_s['sharpe_max']}, 粗糙度={is_s['roughness']}, OOS={oos_s}")
                if a["learning"]:
                    lines.append(f"    教训: {a['learning']}")
        if not self.data["directions"]:
            return "（暂无研究轨迹，这是第一批研究方向）"
        return "\n".join(lines)
