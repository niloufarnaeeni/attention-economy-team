import os
import csv
import ast
import logging

from cmn.team import Team
from cmn.member import Member

log = logging.getLogger(__name__)


class Web3Project(Team):
    """
    Web3Project domain:
    - Loads creators.csv and skills.csv
    - Normalizes project keys (lowercase/strip) to ensure matches
    - Parses messy string lists safely
    - Uses skills.csv.project_id as Team.id (stable ID)
    """

    @staticmethod
    def _parse_list_field(raw):
        """
        Robust parser for list-like fields that may appear as:
        - "['C1','C2']"
        - "[]"
        - "C1"
        - "C1, C2"
        """
        if raw is None:
            return []
        raw = str(raw).strip()
        if not raw:
            return []

        try:
            val = ast.literal_eval(raw)
            if isinstance(val, (list, tuple, set)):
                return [str(x).strip() for x in val if str(x).strip()]
            return [str(val).strip()] if str(val).strip() else []
        except Exception:
            # fallback: split by comma
            s = raw.strip().strip("[]")
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]

    @staticmethod
    def _dedupe_preserve_order(items):
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def read_data(datapath, output, cfg, indexes_only=False):
        # ----------------------------
        # 1) Load project -> creators
        #    (FIXED: use column names)
        # ----------------------------
        project_creators = {}
        creators_path = os.path.join(datapath, "creators.csv")

        with open(creators_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)

            # Optional: fail fast if expected columns are missing
            required = {"project_name", "creators"}
            missing = required - set(r.fieldnames or [])
            if missing:
                raise ValueError(
                    f"creators.csv is missing required columns: {sorted(missing)}. "
                    f"Found columns: {r.fieldnames}"
                )

            for row_idx, row in enumerate(r):
                project_name = (row.get("project_name") or "").strip()
                creator_list = row.get("creators")

                p_key = project_name.lower()
                if not p_key:
                    continue

                creators = Web3Project._parse_list_field(creator_list)
                creators = [str(c).strip().lower() for c in creators if str(c).strip()]
                if creators:
                    project_creators.setdefault(p_key, [])
                    project_creators[p_key].extend(creators)

        # dedupe creators ONCE (after reading file)
        for key in list(project_creators.keys()):
            project_creators[key] = Web3Project._dedupe_preserve_order(project_creators[key])

        # ----------------------------
        # 2) Load project -> (project_id, skills)
        # ----------------------------
        project_skills = {}
        skills_path = os.path.join(datapath, "skills.csv")

        with open(skills_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row_idx, row in enumerate(r):
                project_name = str(row.get("project_name", "")).strip()
                if not project_name:
                    continue
                p_key = project_name.lower()

                pid_raw = row.get("project_id", "")
                if str(pid_raw).strip() == "":
                    continue

                try:
                    project_id = int(pid_raw)
                except Exception:
                    continue

                skill_list = row.get("assigned_skill_ids", "")
                skills = Web3Project._parse_list_field(skill_list)
                skills = [str(s).strip() for s in skills if str(s).strip()]
                if not skills:
                    continue

                if p_key not in project_skills:
                    project_skills[p_key] = {"project_id": project_id, "skills": []}
                else:
                    if project_skills[p_key]["project_id"] != project_id:
                        log.warning(
                            f"[skills.csv] project_name='{project_name}' appears with multiple project_id values: "
                            f"{project_skills[p_key]['project_id']} vs {project_id}. Using the first one."
                        )

                project_skills[p_key]["skills"].extend(skills)

        # dedupe skills
        for key in list(project_skills.keys()):
            project_skills[key]["skills"] = sorted(set(project_skills[key]["skills"]))

        # ----------------------------
        # 3) Build Team objects
        #    IMPORTANT: Team.id = skills.csv.project_id
        # ----------------------------
        teams = {}
        creator2member = {}
        next_member_id = 0
        skipped_no_skill = []

        for row_idx, (p_key, creators) in enumerate(project_creators.items()):
            meta = project_skills.get(p_key)

            if meta is None or not meta.get("skills"):
                skipped_no_skill.append({
                    "row_idx": row_idx,
                    "project_key": p_key,
                    "n_creators": len(creators),
                    "creators_preview": ";".join(creators[:10]),
                })
                continue

            team_id = meta["project_id"]
            skills = meta["skills"]

            if team_id in teams:
                raise ValueError(
                    f"Duplicate project_id detected while building teams: {team_id}. "
                    f"Offending project_key='{p_key}'. Fix skills.csv."
                )

            members = []
            for creator in creators:
                creator = str(creator).strip().lower()
                if not creator:
                    continue

                if creator not in creator2member:
                    m = Member(next_member_id, creator)
                    m.teams = set()
                    m.skills = set()
                    creator2member[creator] = m
                    next_member_id += 1

                m = creator2member[creator]
                m.teams.add(team_id)
                m.skills.update(skills)
                members.append(m)

            if not members:
                continue

            team = Team(
                id=team_id,
                members=members,
                skills=skills,
                datetime="2025",
                location=None,
            )
            teams[team_id] = team

        log.info(
            f"Loaded {len(teams)} projects. Skipped {len(skipped_no_skill)} no-skill projects. "
            f"Unique members: {len(creator2member)}"
        )

        # ----------------------------
        # SHOW skipped projects
        # ----------------------------
        if skipped_no_skill:
            skipped_keys = sorted(set(x["project_key"] for x in skipped_no_skill))

            log.info("========================================")
            log.info("SKIPPED NO-SKILL PROJECTS (%d total):", len(skipped_keys))

            CHUNK = 40
            for i in range(0, len(skipped_keys), CHUNK):
                chunk = skipped_keys[i:i + CHUNK]
                log.info("  %s", ", ".join(chunk))

            log.info("========================================")

        # ----------------------------
        # 4) Delegate to base class
        # ----------------------------
        return Team.read_data(teams, output, cfg)
