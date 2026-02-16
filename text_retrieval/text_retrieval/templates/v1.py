# templates.py

HAS_NEG = False
HAS_SCORE = False

QUERY_TEMPLATE = """Creators suitable for the {project_name} project.

The project requires contributors with the following skill IDs:
{skill_ids}

Skill descriptions:
{skill_descriptions}
"""


DOC_TEMPLATE = """
Creator {creator_id} contributed to the {project_name} project.

Relevant skill IDs for this project:
{skill_ids}
"""

