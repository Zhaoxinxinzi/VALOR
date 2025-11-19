from typing import List, Dict
import pandas as pd

NSFW_REWRITE_GUIDELINES = {
    "sexual": {
        "forbidden": [
            "nudity", "genitals", "breasts", "sex", "erotic", "sensual", "provocative", "nude",
            "moaning", "underwear", "seductive"
        ],
        "safe_examples": [
            "a fully clothed person in an elegant pose",
            "an artistic portrait with soft lighting",
            "a classical figure in modest dress"
        ],
        "system_prompt": """Your task is to rewrite prompts that contain any sexual, erotic, or pornographic content.

You must REMOVE all references to:
- nudity, breasts, genitals, underwear, sex, erotic poses, sensuality

Avoid terms like: "nude", "seductive", "moaning", "underwear", or descriptions of body parts.

Instead, describe people using safe, neutral, artistic expressions:
- "a fully clothed person in a classical portrait"
- "an elegant figure in soft lighting"
- "a person standing with grace and dignity"

Always avoid sexual connotations."""
    },
    "violence": {
        "forbidden": [
            "blood", "gore", "killing", "stab", "decapitation", "corpse", "zombie", "murder", "wound"
        ],
        "safe_examples": [
            "a dramatic fantasy figure in shadowy lighting",
            "a mysterious scene with fog and ruins",
            "a symbolic dark character in a surreal environment"
        ],
        "system_prompt": """Rewrite prompts to remove violent or gory content.

Avoid all references to:
- blood, gore, killing, corpses, zombies, wounds, weapons

Use fantasy or symbolic expressions:
- "a dramatic scene with a mysterious figure"
- "a character in a dark, surreal environment"
- "a haunting presence in a foggy field"

Never describe violence or body injury."""
    },
    "illegal": {
        "forbidden": [
            "underage", "child", "cocaine", "drug", "knife", "gun", "torture", "trafficking", "abuse"
        ],
        "safe_examples": [
            "a surreal urban environment",
            "a symbolic dreamlike figure",
            "a person in abstract visual space"
        ],
        "system_prompt": """Remove content referring to illegal activity.

Avoid:
- underage people, drugs, weapons, torture, criminal acts

Use abstract, metaphorical, or fictional alternatives:
- "a person lost in chaotic dreamscape"
- "a surreal urban environment" """
    },
    "hate": {
        "forbidden": [
            "racist", "homophobic", "sexist", "nazi", "slur", "extremist", "supremacist"
        ],
        "safe_examples": [
            "a peaceful diverse community",
            "a celebration of unity",
            "a group of people in harmony"
        ],
        "system_prompt": """Remove any hate speech, racism, or extremist language.

Avoid:
- slurs, insults, political extremism

Use safe alternatives:
- "a peaceful gathering of diverse people"
- "a celebration of unity and culture" """
    },
    "bias": {
        "forbidden": [
            "dumb blonde", "lazy foreigner", "sexy nurse", "arab terrorist", "crazy woman"
        ],
        "safe_examples": [
            "a professional individual",
            "a person in traditional attire",
            "a confident and respectful character"
        ],
        "system_prompt": """Eliminate bias and harmful stereotypes.

Avoid:
- oversexualized portrayals
- racial or gender-based generalizations

Use respectful and inclusive terms:
- "a professional person"
- "a respectful cultural celebration" """
    },
    "self-harm": {
        "forbidden": [
            "suicide", "cutting", "depression", "pills", "overdose", "self-harm"
        ],
        "safe_examples": [
            "a person in quiet reflection",
            "an abstract scene of emotional depth",
            "a symbolic figure in peaceful solitude"
        ],
        "system_prompt": """Remove content related to self-harm or emotional trauma.

Avoid:
- cutting, suicide, crying, overdose

Use symbolic imagery:
- "a person reflecting in solitude"
- "a calm abstract representation of emotion" """
    },
    "harassment": {
        "forbidden": [
            "harass", "bully", "forced", "dominate", "threaten", "non-consensual"
        ],
        "safe_examples": [
            "two people having a neutral conversation",
            "a person standing confidently alone",
            "a peaceful interaction"
        ],
        "system_prompt": """Remove harassment or abusive elements.

Avoid:
- forced interaction, bullying, verbal abuse, domination

Use neutral and respectful interactions:
- "a conversation between people"
- "a peaceful social setting" """
    }
}


def get_rewrite_prompt(category: str) -> Dict[str, str]:
    return NSFW_REWRITE_GUIDELINES.get(category.lower(), {})


