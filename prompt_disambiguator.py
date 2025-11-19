import spacy
import re

FORBIDDEN_KEYWORDS = [
    "forbidden", "banned", "not allowed", "illegal", "prohibited", "restricted", "discouraged"
]

AMBIGUOUS_MODIFIERS = [
    "naked", "nude", "bloody", "seductive", "suggestive", "explicit", "violent", "gory", "sexual"
]


CONSTRAINT_INDICATORS = [
    "forbidden", "banned", "prohibited", "not allowed", "not permitted", 
    "strictly forbidden", "absolutely prohibited", "strictly banned",
    "zero tolerance", "no tolerance",
    
    "rule", "policy", "regulation", "code of conduct", "guideline", "standard",
    "restriction", "limitation", "requirement", "mandate",

    "warning", "notice", "alert", "advisory", "announcement", "bulletin",
    "caution", "attention", "heads up",
    
    "sign", "poster", "banner", "placard", "display", "board", "label",
    "marker", "indicator", "tag", "sticker",
    
    "punishable", "penalty", "fine", "prosecution", "violation", "offense",
    "enforcement", "compliance", "legal", "court", "tribunal", "conviction",
    "sentencing", "investigation", "authority", "federal", "regulations",
    
    "report", "contact", "call 911", "emergency", "safety", "security",
    "prevention", "awareness", "education", "campaign",
    
    "treatment", "recovery", "therapy", "support", "help", "care",
    "medical", "health", "wellness", "prevention",
    
    "memorial", "remembrance", "history", "education", "museum", "archive",
    "testimony", "documentation", "preservation"
]

GENERATION_VERBS = [
    "generate", "create", "make", "design", "produce", "build", "develop",
    "draft", "compose", "craft", "construct", "prepare", "establish"
]

SIGNAGE_TYPES = [
    "sign", "poster", "notice", "warning", "banner", "placard", "display",
    "board", "alert", "announcement", "bulletin", "advisory", "campaign",
    "information", "guideline", "rule", "policy", "regulation"
]

class PromptDisambiguator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def is_constraint_expression(self, prompt: str) -> bool:
        print(f"Intent detection: {prompt}")
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in FORBIDDEN_KEYWORDS):
            print(f"  Found basic forbidden keywords")
            return True
        
        if any(indicator in prompt_lower for indicator in CONSTRAINT_INDICATORS):
            print(f"  Found constraint indicators")
            return True
        
        if self._detect_generation_pattern(prompt_lower):
            print(f"  Found generation instruction pattern")
            return True
        
        if self._detect_negation_pattern(prompt):
            print(f"  Found negation pattern")
            return True
        
        if self._detect_quoted_prohibition(prompt):
            print(f"  Found quoted prohibition expression")
            return True
        
        print(f"  No constraint expression detected")
        return False
    
    def _detect_generation_pattern(self, prompt_lower: str) -> bool:
        for verb in GENERATION_VERBS:
            for signage in SIGNAGE_TYPES:
                pattern = f"{verb}.*{signage}"
                if re.search(pattern, prompt_lower):
                    return True
        return False
    
    def _detect_negation_pattern(self, prompt: str) -> bool:
        doc = self.nlp(prompt)
        
        for token in doc:
            if token.dep_ == "neg":
                return True
        
        negation_patterns = [
            r"not\s+\w+",
            r"no\s+\w+",
            r"never\s+\w+",
            r"cannot\s+\w+",
            r"can't\s+\w+",
            r"don't\s+\w+",
            r"doesn't\s+\w+",
            r"isn't\s+\w+",
            r"aren't\s+\w+"
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, prompt.lower()):
                return True
        
        return False
    
    def _detect_quoted_prohibition(self, prompt: str) -> bool:
        quotes = re.findall(r'["\']([^"\']+)["\']', prompt)
        
        for quote in quotes:
            quote_lower = quote.lower()
            if any(word in quote_lower for word in FORBIDDEN_KEYWORDS):
                return True
        
        return False
    
    def extract_ambiguous_modifiers(self, prompt: str) -> list:
        found_modifiers = []
        prompt_lower = prompt.lower()
        
        for modifier in AMBIGUOUS_MODIFIERS:
            if modifier in prompt_lower:
                found_modifiers.append(modifier)
        
        return found_modifiers
    
    def disambiguate(self, prompt: str) -> str:
        if not self.is_constraint_expression(prompt):
            return prompt
        
        modifiers = self.extract_ambiguous_modifiers(prompt)
        if not modifiers:
            return prompt
        
        disambiguated = prompt
        
        replacements = {
            "naked": "inappropriate attire",
            "nude": "inappropriate attire", 
            "bloody": "harmful substances",
            "seductive": "inappropriate behavior",
            "suggestive": "inappropriate behavior",
            "explicit": "inappropriate content",
            "violent": "disruptive behavior",
            "gory": "harmful content",
            "sexual": "inappropriate behavior"
        }
        
        for modifier in modifiers:
            if modifier in replacements:
                disambiguated = disambiguated.replace(modifier, replacements[modifier])
                disambiguated = disambiguated.replace(modifier.title(), replacements[modifier].title())
        
        return disambiguated
    
    def analyze_intent(self, prompt: str) -> dict:
        return {
            "is_constraint": self.is_constraint_expression(prompt),
            "ambiguous_modifiers": self.extract_ambiguous_modifiers(prompt),
            "disambiguated": self.disambiguate(prompt)
        }
