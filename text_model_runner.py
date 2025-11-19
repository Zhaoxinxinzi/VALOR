from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
from typing import List, Dict, Tuple
from nsfw_guidelines import get_rewrite_prompt
from prompt_disambiguator import PromptDisambiguator
from sentence_transformers import SentenceTransformer, util
import re
import spacy
import subprocess
import sys
from packaging import version
import random

BLOCKED_TOPICS = [
    "porn", "nude", "naked", "sex", "intercourse", "orgy", "fetish", "erotic",
    "masturbation", "genitals", "vagina", "penis", "boobs", "breasts", "cum",
    "hentai", "NSFW", "lewd", "xxx", "strip", "seduction", "lingerie", "intimate",
    "sexual", "sensual", "provocative", "adult entertainment", "encounters",
    "adult activities", "detailed adult", "sexual activities",

    "rape", "violence", "abuse", "blood", "bloody", "gore", "murder", "torture",
    "stab", "decapitate", "corpse", "zombie", "dismember", "burning body", "killing",
    "assault", "attack", "beating", "shooting", "stabbing", "extremist behavior",

    "underage", "child porn", "loli", "shota", "baby", "schoolgirl", "minor", "teen",
    "grooming", "exploitation", "predatory",

    "suicide", "overdose", "pills", "cutting", "knife", "gun", "weapon", "drug",
    "smoking crack", "injecting heroin", "cocaine", "marijuana", "addict", "bomb",
    "explosive", "firearm", "manufacturing", "recipes", "instructions", "tutorial",
    "weapon manufacturing", "bomb making", "designer drugs", "illegal firearm",

    "fraud", "scam", "robbery", "theft", "hacking", "blackmail", "extortion",
    "trafficking", "smuggling", "kidnapping", "identity theft", "credit card fraud",
    "benefit fraud", "bank fraud", "mail fraud", "strategies", "methods",
    "drug dealing", "human trafficking", "hacking systems", "illegal firearm sales",

    "racist", "slur", "nazi", "homophobic", "sexist", "hate crime", "terrorist",
    "discriminatory", "hate speech", "extremist",

    "harass", "molest", "assault", "coerce", "dominate", "exploit", "non-consensual",

    "eating disorder promotion", "drug dealing tutorials", "bomb making tutorials",
    "weapon manufacturing", "hacking systems", "human trafficking", "extremist behavior",
]

nsfw_refs = [
    "a naked person", "a sexual scene", "a woman showing her breasts", "a pornographic poster",
    "a violent act", "blood all over the floor", "a man stabbing another person",
    "a mutilated body", "a child in a sexual pose", "a girl in lingerie",
    "a zombie covered in blood", "a drug addict injecting heroin",
    "a terrorist holding a bomb", "a depiction of suicide", "a racist sign",
    "a man harassing a woman", "a gory horror creature",
    "a dead body with missing limbs", "someone hanging from a noose",
    "an explicit drawing of genitals", "intimate encounters", "adult entertainment",
    "step-by-step bomb making", "weapon manufacturing guide", "fraud strategies",
    "robbery instructions", "hacking tutorial", "drug recipes"
]

SENSITIVE_LOCATIONS = [
    "congress", "parliament", "white house", "capitol building", "city hall", "courthouse",
    "federal building", "government office", "embassy", "consulate", "pentagon",
    "supreme court", "senate chamber", "house of representatives", "oval office",
    "state house", "town hall", "municipal building", "government building", "state capitol",

    "church", "cathedral", "temple", "mosque", "synagogue", "shrine", "monastery",
    "abbey", "chapel", "basilica", "sacred site", "holy ground",
    "religious sanctuary", "house of worship", "prayer hall", "sacred space",

    "holocaust memorial", "war memorial", "veterans memorial", "9/11 memorial",
    "genocide memorial", "memorial park", "remembrance site", "cemetery",
    "national monument", "tomb of unknown soldier", "memorial wall", "memorial plaza",
    "commemorative site", "historical monument", "martyrs memorial", "monument",

    "school", "university", "college", "kindergarten", "elementary school", "high school",
    "classroom", "library", "lecture hall", "campus", "educational facility",

    "hospital", "clinic", "emergency room", "medical center", "nursing home",
    "intensive care unit", "medical facility", "health center",

    "police station", "jail", "prison", "detention center", "courtroom", "legal office"
]

INAPPROPRIATE_ACTS = [
    "pole dancing", "strip dancing", "lap dancing", "twerking", "erotic dancing",
    "nude", "nudity", "naked", "streaking", "sexual behavior", "intimate activities",
    "making out", "sexual poses", "seductive behavior", "sexual acts", "sexual activities",

    "fighting", "brawling", "physical altercation", "aggressive behavior", "assault",
    "battery", "violent", "violence", "physical violence", "beating", "attack",

    "mocking", "ridiculing", "making jokes", "joking", "inappropriate humor",
    "offensive comedy", "parody", "disrespectful", "insulting",

    "drinking alcohol", "consuming drugs", "smoking", "drug use", "alcohol consumption",
    "substance abuse", "intoxicated", "drunken", "getting drunk",

    "loud music", "shouting", "screaming", "noise making", "party noise",
    "disruptive behavior", "unruly behavior", "inappropriate celebration"
]

ENTERTAINMENT_INDICATORS = [
    "for fun", "for entertainment", "for shock", "for kicks", "as a joke",
    "to be provocative", "for viral content", "for social media", "for laughs",
    "just for fun", "for amusement", "for thrills", "for shock value",
    "as prank", "for meme", "for a laugh", "for joke", "for fun only", "just for entertainment", "for prank", "for tiktok", "for instagram", "for youtube", "for twitter", "for facebook"
]

STRICT_CONSTRAINT_PATTERNS = [
    r"['\"](.*?(forbidden|prohibited|banned|not\s+allowed|illegal|penalty|prosecution|fine).*?)['\"]",

    r"(generate|create|make|design)\s+.*?(sign|poster|notice|warning|restriction).*?['\"](.*?)['\"]",

    r"(health|safety|legal|medical|educational|information|awareness|prevention)\s+(education|information|campaign|poster|notice)",

    r"(memorial|historical|commemoration)\s+(education|documentation|display|information)",

    r"(prohibition sign|official documentation|restriction notice|policy notice|warning sign)",
    r"(generate|create|make|design)\s+.*?(prohibition sign|notice|official documentation|restriction|policy|warning sign)"
]

FURTHER_IMPROVED_GOOD_PATTERNS = [
    r"(generate|create|make|design).*?(restriction|warning|prohibition|notice|policy).*?['\"](.*?(forbidden|prohibited|banned|not\s+allowed|penalty|prosecution).*?)['\"]",

    r"(health|medical|safety|educational)\s+(education|information|awareness|prevention|campaign|documentation|material)",

    r"(memorial|historical|genocide|holocaust)\s+(education|documentation|information|awareness)",

    r"(professional|official|diplomatic)\s+(tour|visit|ceremony|event|documentation)",

    r"(legal|policy|official)\s+(information|documentation|notice|guidelines)",

    r"(create|generate|make)\s+(professional|official)\s+(documentation|information|material)",

    r"(create|generate|make)\s+(educational|health)\s+(material|content|information)"
]

FURTHER_IMPROVED_BAD_PATTERNS = [
    r".*(for\s+fun|for\s+entertainment|for\s+shock|for\s+kicks|as\s+a\s+joke|to\s+be\s+provocative|for\s+social\s+media|for\s+viral\s+content|for\s+amusement|for\s+thrills)",

    r"(hot|sexy|beautiful|attractive).*?(women|girls|models).*(pole\s+dancing|strip|naked|sexual|erotic|twerking)",

    r"(show|display|depict|create|make|generate).*(violence|sexual|naked|fighting|beating|drinking|drug)",

    r"(people|group|crowd|persons|individuals).*(naked|sexual|violent|fighting|drinking|drug|disruptive)",

    r"(tutorial|guide|instructions|how\s+to).*(bomb|weapon|drug|illegal)",

    r".*(congress|parliament|church|temple|memorial|school|hospital|police\s+station).*(naked|sexual|fighting|drinking|disruptive|twerking|pole\s+dancing)",

    r".*(for\s+fun|for\s+entertainment|for\s+social\s+media).*(congress|parliament|church|temple|memorial|school|hospital)"
]

FURTHER_IMPROVED_INTENTION_ANALYSIS_PROMPT = """You are a strict intent classifier for content safety. \
Your job is to judge if the following prompt is for educational/professional/warning (GOOD) or for entertainment/harmful/irreverent (BAD).\n\nCRITICAL RULES:\n- If the prompt contains entertainment intent (e.g. 'for fun', 'for entertainment', 'for shock', 'for social media'), sexualized or disrespectful behavior in sensitive locations (e.g. 'genocide memorial', 'church', 'school'), or describes inappropriate acts (e.g. 'streaking', 'drinking', 'fighting'), always respond BAD.\n- If the prompt is about creating a warning sign, educational material, or official notice, respond GOOD.\n- Only output 'GOOD: <reason>' or 'BAD: <reason>'.\n\nEXAMPLES:\n'Generate notice: violence prohibited in school' => GOOD: warning sign creation\n'People streaking in genocide memorial for fun' => BAD: inappropriate behavior in sensitive location for entertainment\n'Create health education poster about drug abuse' => GOOD: educational\n'Hot women pole dancing in parliament' => BAD: sexualized behavior in sensitive location\n'People drinking for fun at memorial' => BAD: substance abuse in sensitive location for entertainment\n'Create official policy notice about safety' => GOOD: official informational content\n'Generate restriction sign: "no smoking in hospital"' => GOOD: prohibition sign creation\n'People fighting in classroom for fun' => BAD: violence in educational institution for entertainment\n\nAnalyze: {prompt}"""

FURTHER_IMPROVED_VALUE_DETECTION_PROMPT = """You are a strict value safety classifier for image generation prompts.
If the prompt contains any entertainment, disrespect, inappropriate, sexual, violent, or mocking behavior in sensitive locations, always respond HARMFUL.
If the prompt is a warning, prohibition, official notice, educational, health, legal, or historical content, respond SAFE.
Only output one line: SAFE: [reason] or HARMFUL: [reason]
Prompt: {prompt}
"""

SYSTEM_PROMPT = """You are an AI assistant that rewrites prompts for image generation.

You must make the prompt SAFE by removing all harmful or unsafe content, including:
- ❌ Sexual / nudity / erotic / intimate content
- ❌ Violence / gore / horror / weapons
- ❌ Hate / racism / bias / discrimination
- ❌ Drugs / substance abuse / illegal activities
- ❌ Self-harm / suicide / dangerous behavior
- ❌ Disrespectful behavior in sensitive locations
- ❌ Entertainment intent in inappropriate contexts

You should:
- ✅ Keep the core visual elements and artistic style
- ✅ Maintain the overall composition and scene
- ✅ Preserve professional, educational, or informational intent
- ✅ Use respectful, neutral language
- ✅ Focus on safe, appropriate content
- ✅ Replace harmful elements with safe alternatives

Rewrite the prompt to be completely safe while maintaining visual quality and artistic intent."""

VALUE_PROMPT =  """You are an expert in cultural appropriateness assessment for AI-generated content. Evaluate content for potential cultural sensitivity issues and provide contextually appropriate alternatives when necessary.

EVALUATION FRAMEWORK:
Context Assessment: Identify the setting type (religious, governmental, memorial, educational, cultural, healthcare)
Behavioral Analysis: Assess activities for contextual appropriateness based on established social norms
Cultural Impact: Consider potential cultural or social sensitivities

CONTEXTUAL APPROPRIATENESS FACTORS:
- Alignment with location-specific behavioral expectations
- Respect for cultural and religious practices
- Consideration of historical significance and solemnity
- Educational and professional environment standards
- Public decorum and social conventions

MODIFICATION APPROACH:
When content presents potential sensitivity concerns:
- Suggest alternative contexts that preserve creative intent
- Recommend behavioral modifications maintaining respect
- Provide culturally appropriate alternatives
- Maintain content quality while addressing concerns  """

INTENTION_PROMPT = """You are an assistant specializing in prompt optimization for AI-generated visual communication systems. 

OBJECTIVE: Transform user input into specifications for standardized prohibition signage adhering to established visual communication principles.

DESIGN PARAMETERS:
- Visual Foundation: Neutral background for maximum readability
- Typography: Bold, high-contrast text following legibility standards  
- Design Framework: Professional signage conventions consistent with regulatory standards
- Symbolic Integration: Universal prohibition symbols (negation indicators, crossed-out iconography)
- Layout: Clean, hierarchical information architecture prioritizing message clarity

IMPLEMENTATION APPROACH:
Generate visual representations of prohibition concepts rather than depicting prohibited activities. 
Focus on the communicative function of signage as regulatory visual artifacts, emphasizing symbolic representation over literal depiction."""

VISUAL_ENHANCEMENT_PATTERNS = {
    "restriction": {
        "keywords": ["restriction", "not permitted", "forbidden", "banned", "prohibited"],
        "suffix": " - clear prohibition sign with red circle and diagonal line, simple white background, bold text, official warning sign design"
    },
    "warning": {
        "keywords": ["warning", "danger", "caution", "alert"],
        "suffix": " - professional warning sign with clean design, clear prohibition symbol, white background, regulatory style"
    },
    "notice": {
        "keywords": ["notice", "announcement", "advisory", "bulletin"],
        "suffix": " - official notice board with clear text, professional layout, white background, authoritative design"
    },
    "health": {
        "keywords": ["health", "medical", "prevention", "safety", "wellness"],
        "suffix": " - health education poster with professional medical design, clear informational layout, clean typography"
    },
    "memorial": {
        "keywords": ["memorial", "remembrance", "historical", "genocide", "commemoration"],
        "suffix": " - respectful memorial display with dignified design, solemn presentation, appropriate cultural sensitivity"
    },
    "educational": {
        "keywords": ["education", "awareness", "campaign", "information", "learning"],
        "suffix": " - educational poster with clear informational design, professional layout, accessible typography"
    },
    "legal": {
        "keywords": ["legal", "law", "statute", "justice", "court", "prosecution"],
        "suffix": " - legal information poster with official government design, formal layout, authoritative appearance"
    },
    "safety": {
        "keywords": ["safety", "crime", "report", "community", "protection"],
        "suffix": " - community safety poster with clear call-to-action design, professional emergency services style"
    },
    "general_safe": {
        "keywords": ["safe", "appropriate", "artistic"],
        "suffix": " - in artistic illustration style, safe and respectful composition, professional quality"
    }
}

CONSTRAINT_REWRITE_PATTERNS = {
    "knife": "sharp objects",
    "gun": "weapons",
    "weapon": "prohibited items",
    "brandishing": "displaying",
    "violent": "disruptive",
    "assault": "inappropriate contact",
    "attack": "aggressive behavior",
    "blood": "harmful substances",
    "naked": "inappropriate attire",
    "nude": "improper dress",
    "sexual": "inappropriate behavior",
    "drug": "controlled substances",
    "cocaine": "illegal substances",
    "marijuana": "prohibited substances",
    "heroin": "controlled substances",
    "abuse": "misconduct",
    "harassment": "inappropriate behavior",
    "fraud": "illegal activities",
    "bomb": "prohibited devices",
    "explosive": "dangerous materials",
    "intimate": "inappropriate",
    "erotic": "inappropriate",
    "pornographic": "inappropriate",
}


class TextModelRewriter:
    def __init__(self, model_type: str = "gpt2", model_path: str = None):
        self.model_type = model_type.lower()
        self.model_path = model_path or self.get_default_path(self.model_type)
        self.tokenizer, self.model = self.load_model()
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        self.disambiguator = PromptDisambiguator()
        self.embed_model = SentenceTransformer('file_path/all-MiniLM-L6-v2')
        self.spacy_nlp = spacy.load("en_core_web_sm")

        self._detection_cache = {}

    def get_default_path(self, model_type: str) -> str:
        default_paths = {
            "zephyr": "file_path/Zephyr-7B-Beta",
            "qwen": "file_path/Qwen1.5-1.8B-Chat",
            "deepseek": "file_path/deepseek-llm-7b-chat",
            "llama": "file_path/llama-3.2-3B-instruct"
        }
        return default_paths.get(model_type, default_paths["deepseek"])

    def load_model(self):
        print(f"Loading LLM model: {self.model_type} from {self.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            ).eval()
            print("Model loaded successfully (compatibility mode)")
        except Exception as e:
            print(f"Trying CPU mode: {e}")
            model_kwargs["device_map"] = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            ).eval()
            print("Model loaded successfully (CPU mode)")

        return tokenizer, model

    def format_prompt_for_model(self, system_prompt: str, user_prompt: str, model_type: str) -> str:
        if "deepseek" in model_type.lower():
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        elif "qwen" in model_type.lower():
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        elif "llama" in model_type.lower():
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] "

        else:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "

    def safe_tokenizer_encode(self, text: str, model_type: str):
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": text}
                ]
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = self.tokenizer(
                        formatted_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )
                    return inputs
                except:
                    pass

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
                add_special_tokens=True
            )
            return inputs

        except Exception as e:
            print(f"       Tokenizer encoding failed: {e}")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            return inputs

    def get_generation_kwargs(self, model_type: str) -> dict:
        """Get model-specific generation parameters"""

        base_kwargs = {
            "max_new_tokens": 80,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if "qwen" in model_type.lower():
            return {
                **base_kwargs,
                "repetition_penalty": 1.1,
            }
        elif "deepseek" in model_type.lower():
            return {
                **base_kwargs,
                "temperature": 0.5,
            }
        else:
            return base_kwargs

    def query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Improved LLM query method"""

        print(f"       Calling {self.model_type} model...")

        try:
            # 1. Format prompt according to model type
            formatted_text = self.format_prompt_for_model(system_prompt, user_prompt, self.model_type)

            # 2. Safe tokenizer encoding
            inputs = self.safe_tokenizer_encode(formatted_text, self.model_type)

            # 3. Device handling
            try:
                if torch.cuda.is_available() and hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
            except Exception as e:
                print(f"       GPU transfer failed, using CPU: {e}")
                if hasattr(self.model, 'cpu'):
                    self.model = self.model.cpu()

            # 4. Model-specific generation parameters
            generation_kwargs = self.get_generation_kwargs(self.model_type)

            with torch.no_grad():
                try:
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.get('attention_mask', None),
                        **generation_kwargs
                    )
                except Exception as gen_error:
                    print(f"       Generation failed, trying simplified parameters: {gen_error}")
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # 5. Decode output
                new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
                reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                print(f"       LLM generation succeeded: '{reply[:50]}...'")
                return reply

        except Exception as e:
            print(f"       LLM query failed: {e}")
            print(f"       Using rule-based fallback...")
            return self._fallback_detection(system_prompt, user_prompt)

    def is_strict_constraint_expression(self, prompt: str) -> bool:
        """Strict constraint expression detection - reduce false positives"""

        print(f"  Strict constraint expression detection: {prompt}")

        # 1. Must have quoted prohibition expressions
        quotes_content = re.findall(r'["\']([^"\']+)["\']', prompt)
        prohibition_words = ["forbidden", "prohibited", "banned", "not allowed", "illegal", "penalty", "prosecution"]

        for quote in quotes_content:
            quote_lower = quote.lower()
            for prohibition in prohibition_words:
                if prohibition in quote_lower:
                    print(f"    Quoted prohibition expression: '{quote}' contains '{prohibition}'")
                    return True

        # 2. Strict pattern matching
        for pattern in STRICT_CONSTRAINT_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                print(f"    Strict constraint pattern: {pattern[:30]}...")
                return True

        # 3. Good intention pattern detection
        for pattern in FURTHER_IMPROVED_GOOD_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                print(f"    Clear good intention pattern: {pattern[:30]}...")
                return True

        print(f"    Failed strict constraint expression detection")
        return False

    def improved_intention_detection(self, prompt: str) -> Tuple[str, str]:
        """Improved intention detection - prioritize LLM, rule-based fallback"""
        prompt_lower = prompt.lower()
        print(f"  Improved intention detection (LLM priority): {prompt}")

        # 1. Prioritize LLM judgment
        try:
            llm_response = self.query_llm(FURTHER_IMPROVED_INTENTION_ANALYSIS_PROMPT.format(prompt=prompt), "")
            print(f"    LLM intention judgment result: {llm_response}")
            response_clean = llm_response.strip().lower()
            if response_clean.startswith("good"):
                intention = "GOOD"
                reason = llm_response
                print(f"    LLM judged as GOOD")
                return intention, reason
            elif response_clean.startswith("bad"):
                intention = "BAD"
                reason = llm_response
                print(f"    LLM judged as BAD")
                return intention, reason
        except Exception as e:
            print(f"    LLM intention judgment failed: {e}")

        # 2. Rule-based fallback
        print(f"    LLM unable to judge, using rule-based fallback")
        return "BAD", "Both LLM and rules unable to judge, fallback to BAD"

    def enhanced_word_level_detection(self, prompt: str) -> Tuple[bool, str]:
        prompt_lower = prompt.lower()
        # 1. Prioritize compliance expression exemption
        compliance_keywords = [
            "prohibition sign", "notice", "documentation", "education", "warning", "prevention", "health", "safety", "policy", "guideline", "historical", "memorial", "restriction", "penalty", "regulation", "rule", "official", "professional"
        ]
        compliance_verbs = ["generate", "create", "make", "design", "produce"]
        if any(kw in prompt_lower for kw in compliance_keywords) and any(vb in prompt_lower for vb in compliance_verbs):
            return False, "compliance expression exemption"
        # 2. Sensitive location + inappropriate behavior combination, directly harmful
        sensitive_locations_412 = [
            'congress', 'parliament', 'government', 'senate', 'house', 'capitol', 'capitol building', 'capitol hill',
            'church', 'temple', 'mosque', 'synagogue', 'cathedral', 'shrine', 'basilica', 'religious sanctuary', 'house of worship',
            'memorial', 'cemetery', 'grave', 'tomb', 'monument', 'memorial park', 'memorial wall', 'memorial plaza', 'genocide memorial', 'holocaust memorial', '9/11 memorial', 'veterans memorial', 'tomb of unknown soldier',
            'school', 'university', 'college', 'academy', 'institution', 'elementary school', 'kindergarten', 'campus', 'lecture hall', 'classroom',
            'hospital', 'clinic', 'medical', 'healthcare', 'emergency', 'emergency room', 'intensive care unit', 'medical center', 'nursing home',
            'police station', 'court', 'judicial', 'legal', 'law enforcement', 'courthouse', 'courtroom', 'legal office', 'judicial building',
            'white house', 'oval office', 'federal building', 'embassy', 'consulate', 'city hall', 'town hall', 'community center', 'cultural center', 'art gallery', 'museum', 'library', 'fire station', 'landmark',
            'mosque', 'shrine', 'basilica', 'prayer room', 'holy site', 'sacred ground'
        ]
        inappropriate_acts_412 = [
            'dancing', 'pole dancing', 'strip dancing', 'lap dancing', 'erotic dancing', 'naked', 'nude', 'nudity', 'sexual', 'erotic', 'adult', 'sexual acts', 'sexual behavior', 'sexual poses', 'making out', 'intimate activities',
            'violent', 'fighting', 'beating', 'assault', 'attack', 'rioting', 'brawling', 'aggressive behavior', 'physical violence', 'violent confrontation', 'battery',
            'party', 'drinking', 'alcohol', 'drug', 'smoking', 'getting drunk', 'drunken behavior', 'intoxicated behavior', 'consuming drugs', 'substance abuse',
            'vandalizing', 'twerking', 'stripping', 'posing', 'seductive', 'provocative', 'streaking',
            'disruptive behavior', 'unruly behavior', 'inappropriate celebration', 'shouting', 'screaming', 'noise making', 'loud music',
            'satirical performance', 'inappropriate humor', 'making jokes', 'ridiculing', 'mocking', 'parody', 'offensive comedy',
            'business promotion', 'commercial advertising', 'vending', 'selling products',
            'flashing', 'moon', 'exposing', 'public urination', 'public sex', 'public nudity'
        ]
        if any(loc in prompt_lower for loc in sensitive_locations_412) and any(act in prompt_lower for act in inappropriate_acts_412):
            return True, "sensitive location with inappropriate behavior"
        # 3. Entertainment intent + sensitive location
        entertainment_indicators_412 = [
            "for fun", "for entertainment", "for shock", "for kicks", "for social media", "for viral content", "for amusement", "for thrills", "to be provocative", "as a joke", "for laughs", "for the lulz", "for memes", "for tiktok", "for instagram", "for youtube", "for twitter", "for facebook", "just for fun", "for shock value", "for joke", "for fun only", "just for entertainment", "for meme", "for prank", "for a laugh"
        ]
        if any(loc in prompt_lower for loc in sensitive_locations_412) and any(ent in prompt_lower for ent in entertainment_indicators_412):
            return True, "entertainment intent with sensitive location"
        # 4. Precise sensitive word detection
        rule_based_result = self.contains_blocked_words_raw(prompt)
        found_words = []
        if rule_based_result:
            for word in BLOCKED_TOPICS:
                if word.lower() in prompt_lower:
                    found_words.append(word)
            print(f"     Rule-based detection found sensitive words: {found_words[:5]}")

        # 5. Precise entertainment intent detection
        entertainment_detected = False
        entertainment_reason = ""
        
        # Detect entertainment intent keywords
        for indicator in entertainment_indicators_412:
            if indicator in prompt_lower:
                entertainment_detected = True
                entertainment_reason = f"entertainment intent: {indicator}"
                break
        
        # 6. Precise sexualized content detection
        sexualized_patterns_412 = [
            r"(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*?(women|girls|models|ladies|females).*(pole\\s+dancing|strip|naked|nude|sexual|erotic|twerking|dancing|posing|streaking|lap\\s+dancing)",
            r"(women|girls|models|ladies|females).*(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*(pole\\s+dancing|strip|naked|nude|sexual|erotic|twerking|dancing|posing|streaking|lap\\s+dancing)",
            r"(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*?(women|girls|models|ladies|females)",
            r"(women|girls|models|ladies|females).*(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring)",
            r"(bikini\\s+models|young\\s+women|two\\s+young\\s+women|hot\\s+girls).*(sexual|erotic|naked|nude|streaking|twerking|pole\\s+dancing|lap\\s+dancing)",
            r"(lingerie|underwear|bikini|swimsuit).*(dancing|posing|stripping|twerking)",
            r"(stripper|exotic dancer|burlesque).*"
        ]
        for pattern in sexualized_patterns_412:
            if re.search(pattern, prompt_lower):
                entertainment_detected = True
                entertainment_reason = "sexualized content with inappropriate behavior"
                break
        
        # 7. Precise harmful content detection
        harmful_patterns_412 = [
            r"(show|display|depict|create|make|generate|produce).*(violence|violent|blood|bloody|gore|murder|torture|assault|attack|fighting|beating|shooting|stabbing|killing|physical\\s+violence|violent\\s+confrontation)",
            r"(people|group|crowd|persons|individuals|men|women|guys|girls|someone|two|delegates|officials|employees|teenagers|students).*(naked|nude|sexual|erotic|adult|intimate|breast|genital|violent|fighting|beating|assault|attack|rioting|drinking|alcohol|drug|smoking|disruptive|vandalizing|twerking|stripping|posing|seductive|provocative|streaking|lap\\s+dancing|pole\\s+dancing|brawling|aggressive\\s+behavior|inappropriate\\s+celebration|disruptive\\s+behavior|unruly\\s+behavior|shouting|screaming|noise\\s+making|getting\\s+drunk|drunken\\s+behavior|intoxicated\\s+behavior|consuming\\s+drugs|satirical\\s+performance|inappropriate\\s+humor|making\\s+jokes|ridiculing|mocking|parody|offensive\\s+comedy|business\\s+promotion|commercial\\s+advertising|vending|selling\\s+products|loud\\s+music)",
            r"(show|display|depict|create|make|generate|produce).*(sexual|naked|nude|erotic|porn|adult|intimate|breast|genital|sexual\\s+acts|sexual\\s+behavior|sexual\\s+poses|making\\s+out|intimate\\s+activities)",
            r"(naked|nude|sexual|erotic|adult|intimate|breast|genital|sexual\\s+acts|sexual\\s+behavior|sexual\\s+poses|making\\s+out|intimate\\s+activities).*(women|girls|models|ladies|females|men|guys|males)",
            r"(riot|chaos|mayhem|anarchy|looting|arson|vandalism)",
            r"(self-harm|suicide|overdose|cutting|slashing|injecting drugs)"
        ]
        for pattern in harmful_patterns_412:
            if re.search(pattern, prompt_lower):
                entertainment_detected = True
                entertainment_reason = "showing/displaying harmful content"
                break

        # 8. If no sensitive words but entertainment intent, mark as harmful
        if not found_words and entertainment_detected:
            print(f"     Entertainment intent detected: {entertainment_reason}")
            return True, entertainment_reason

        # 9. No sensitive words and no entertainment intent, pass directly
        if not found_words and not entertainment_detected:
            print(f"     No sensitive words or entertainment intent, word detection passed")
            return False, "no sensitive words or entertainment intent detected"

        # 10. When sensitive words are present, only very strict constraint expressions are exempt
        if found_words and self.is_strict_constraint_expression(prompt):
            # Additional check: ensure it's really educational/informational content
            educational_indicators_412 = ["education", "information", "warning", "notice", "prevention", "health", "safety", "policy", "documentation", "professional", "official", "legal", "guidelines", "historical", "memorial", "restriction", "prohibition", "penalty", "consequences", "regulation", "rule"]
            has_educational = any(word in prompt_lower for word in educational_indicators_412)
            has_quotes = '"' in prompt or "'" in prompt
            has_creation_verbs = any(verb in prompt_lower for verb in ["generate", "create", "make", "design", "produce"])
            
            if has_educational and (has_quotes or has_creation_verbs):
                print(f"     Strict constraint expression exemption: sensitive words in sign creation context")
                return False, f"strict constraint expression with words: {found_words[:3]}"
            else:
                print(f"     Constraint expression not strict enough, marked as harmful")
                return True, f"insufficient constraint expression with words: {found_words[:3]}"

        if found_words:
            print(f"     Sensitive words found and not strict constraint expression, marked as harmful")
            return True, f"harmful words detected: {found_words[:3]}"
        else:
            print(f"     Entertainment intent detected harmful content")
            return True, entertainment_reason

        for indicator in ENTERTAINMENT_INDICATORS:
            for location in SENSITIVE_LOCATIONS:
                if indicator in prompt_lower and location in prompt_lower:
                    return True, f"entertainment intent '{indicator}' in sensitive location '{location}'"

    def enhanced_semantic_level_detection(self, prompt: str) -> Tuple[bool, str]:
        if self.is_strict_constraint_expression(prompt):
            prompt_lower = prompt.lower()
            educational_indicators_412 = ["education", "information", "warning", "notice", "prevention", "health", "safety", "policy", "documentation", "professional", "official", "legal", "guidelines", "historical", "memorial", "restriction", "prohibition", "penalty", "consequences", "regulation", "rule"]
            has_educational = any(word in prompt_lower for word in educational_indicators_412)
            has_quotes = '"' in prompt or "'" in prompt
            has_creation_verbs = any(verb in prompt_lower for verb in ["generate", "create", "make", "design", "produce"])
            
            if has_educational and (has_quotes or has_creation_verbs):
                print(f"     Strict constraint expression exemption: semantically safe")
                return False, "strict constraint expression - semantically safe"
            else:
                print(f"     Constraint expression not strict enough, continue detection")

        semantic_score = self.get_semantic_similarity_score(prompt)
        threshold = 0.32

        if semantic_score > threshold:
            print(f"     Semantic similarity too high: {semantic_score:.3f} > {threshold}")
            return True, f"high semantic similarity: {semantic_score:.3f}"

        prompt_lower = prompt.lower()
        
        entertainment_indicators_412 = [
            "for fun", "for entertainment", "for shock", "for kicks", "for social media", 
            "for viral content", "for amusement", "for thrills", "to be provocative",
            "as a joke", "for laughs", "for the lulz", "for memes",
            "for tiktok", "for instagram", "for youtube", "for twitter", "for facebook",
            "just for fun", "for shock value", "to be provocative",
            "for joke", "for fun only", "just for entertainment", "for meme", "for prank", "for a laugh"
        ]
        
        has_entertainment = any(indicator in prompt_lower for indicator in entertainment_indicators_412)
        
        sensitive_locations_412 = [
            'congress', 'parliament', 'government', 'senate', 'house', 'capitol', 'capitol building', 'capitol hill',
            'church', 'temple', 'mosque', 'synagogue', 'cathedral', 'shrine', 'basilica', 'religious sanctuary', 'house of worship',
            'memorial', 'cemetery', 'grave', 'tomb', 'monument', 'memorial park', 'memorial wall', 'memorial plaza', 'genocide memorial', 'holocaust memorial', '9/11 memorial', 'veterans memorial', 'tomb of unknown soldier',
            'school', 'university', 'college', 'academy', 'institution', 'elementary school', 'kindergarten', 'campus', 'lecture hall', 'classroom',
            'hospital', 'clinic', 'medical', 'healthcare', 'emergency', 'emergency room', 'intensive care unit', 'medical center', 'nursing home',
            'police station', 'court', 'judicial', 'legal', 'law enforcement', 'courthouse', 'courtroom', 'legal office', 'judicial building',
            'white house', 'oval office', 'federal building', 'embassy', 'consulate', 'city hall', 'town hall', 'community center', 'cultural center', 'art gallery', 'museum', 'library', 'fire station', 'landmark',
            'mosque', 'shrine', 'basilica', 'prayer room', 'holy site', 'sacred ground'
        ]
        
        inappropriate_acts_412 = [
            'dancing', 'pole dancing', 'strip dancing', 'lap dancing', 'erotic dancing', 'naked', 'nude', 'nudity', 'sexual', 'erotic', 'adult', 'sexual acts', 'sexual behavior', 'sexual poses', 'making out', 'intimate activities',
            'violent', 'fighting', 'beating', 'assault', 'attack', 'rioting', 'brawling', 'aggressive behavior', 'physical violence', 'violent confrontation', 'battery',
            'party', 'drinking', 'alcohol', 'drug', 'smoking', 'getting drunk', 'drunken behavior', 'intoxicated behavior', 'consuming drugs', 'substance abuse',
            'vandalizing', 'twerking', 'stripping', 'posing', 'seductive', 'provocative', 'streaking',
            'disruptive behavior', 'unruly behavior', 'inappropriate celebration', 'shouting', 'screaming', 'noise making', 'loud music',
            'satirical performance', 'inappropriate humor', 'making jokes', 'ridiculing', 'mocking', 'parody', 'offensive comedy',
            'business promotion', 'commercial advertising', 'vending', 'selling products',
            'flashing', 'moon', 'exposing', 'public urination', 'public sex', 'public nudity'
        ]
        
        has_sensitive_location = any(loc in prompt_lower for loc in sensitive_locations_412)
        has_inappropriate_act = any(act in prompt_lower for act in inappropriate_acts_412)
        
        if has_entertainment and (has_sensitive_location or has_inappropriate_act):
            print(f"     Entertainment intent with sensitive content combination detection")
            return True, "entertainment intent with sensitive content"

        sexualized_patterns_412 = [
            r"(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*?(women|girls|models|ladies|females)",
            r"(women|girls|models|ladies|females).*(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring)",
            r"(bikini\\s+models|young\\s+women|two\\s+young\\s+women|hot\\s+girls).*(sexual|erotic|naked|nude|streaking|twerking|pole\\s+dancing|lap\\s+dancing)",
            r"(lingerie|underwear|bikini|swimsuit).*(dancing|posing|stripping|twerking)",
            r"(stripper|exotic dancer|burlesque).*"
        ]
        for pattern in sexualized_patterns_412:
            if re.search(pattern, prompt_lower):
                print(f"     Sexualized content detection")
                return True, "sexualized content detected"

        harmful_patterns_412 = [
            r"(show|display|depict|create|make|generate|produce).*(violence|violent|blood|bloody|gore|murder|torture|assault|attack|fighting|beating|shooting|stabbing|killing|physical\\s+violence|violent\\s+confrontation)",
            r"(people|group|crowd|persons|individuals|men|women|guys|girls|someone|two|delegates|officials|employees|teenagers|students).*(naked|nude|sexual|erotic|adult|intimate|breast|genital|violent|fighting|beating|assault|attack|rioting|drinking|alcohol|drug|smoking|disruptive|vandalizing|twerking|stripping|posing|seductive|provocative|streaking|lap\\s+dancing|pole\\s+dancing|brawling|aggressive\\s+behavior|inappropriate\\s+celebration|disruptive\\s+behavior|unruly\\s+behavior|shouting|screaming|noise\\s+making|getting\\s+drunk|drunken\\s+behavior|intoxicated\\s+behavior|consuming\\s+drugs|satirical\\s+performance|inappropriate\\s+humor|making\\s+jokes|ridiculing|mocking|parody|offensive\\s+comedy|business\\s+promotion|commercial\\s+advertising|vending|selling\\s+products|loud\\s+music)",
            r"(show|display|depict|create|make|generate|produce).*(sexual|naked|nude|erotic|porn|adult|intimate|breast|genital|sexual\\s+acts|sexual\\s+behavior|sexual\\s+poses|making\\s+out|intimate\\s+activities)",
            r"(naked|nude|sexual|erotic|adult|intimate|breast|genital|sexual\\s+acts|sexual\\s+behavior|sexual\\s+poses|making\\s+out|intimate\\s+activities).*(women|girls|models|ladies|females|men|guys|males)",
            r"(riot|chaos|mayhem|anarchy|looting|arson|vandalism)",
            r"(self-harm|suicide|overdose|cutting|slashing|injecting drugs)"
        ]
        for pattern in harmful_patterns_412:
            if re.search(pattern, prompt_lower):
                print(f"     Harmful content detection")
                return True, "harmful content detected"

        print(f"     Semantic detection passed: {semantic_score:.3f} <= {threshold}")
        return False, f"semantic similarity acceptable: {semantic_score:.3f}"

    def improved_value_level_detection(self, prompt: str) -> Tuple[bool, str]:
        prompt_lower = prompt.lower()
        compliance_keywords = [
            "prohibition sign", "notice", "documentation", "education", "warning", "prevention", "health", "safety", "policy", "guideline", "historical", "memorial", "restriction", "penalty", "regulation", "rule", "official", "professional"
        ]
        compliance_verbs = ["generate", "create", "make", "design", "produce"]
        if any(kw in prompt_lower for kw in compliance_keywords) and any(vb in prompt_lower for vb in compliance_verbs):
            return False, "compliance expression exemption"
        sensitive_locations_412 = [
            'congress', 'parliament', 'government', 'senate', 'house', 'capitol', 'capitol building', 'capitol hill',
            'church', 'temple', 'mosque', 'synagogue', 'cathedral', 'shrine', 'basilica', 'religious sanctuary', 'house of worship',
            'memorial', 'cemetery', 'grave', 'tomb', 'monument', 'memorial park', 'memorial wall', 'memorial plaza', 'genocide memorial', 'holocaust memorial', '9/11 memorial', 'veterans memorial', 'tomb of unknown soldier',
            'school', 'university', 'college', 'academy', 'institution', 'elementary school', 'kindergarten', 'campus', 'lecture hall', 'classroom',
            'hospital', 'clinic', 'medical', 'healthcare', 'emergency', 'emergency room', 'intensive care unit', 'medical center', 'nursing home',
            'police station', 'court', 'judicial', 'legal', 'law enforcement', 'courthouse', 'courtroom', 'legal office', 'judicial building',
            'white house', 'oval office', 'federal building', 'embassy', 'consulate', 'city hall', 'town hall', 'community center', 'cultural center', 'art gallery', 'museum', 'library', 'fire station', 'landmark',
            'mosque', 'shrine', 'basilica', 'prayer room', 'holy site', 'sacred ground'
        ]
        inappropriate_acts_412 = [
            'dancing', 'pole dancing', 'strip dancing', 'lap dancing', 'erotic dancing', 'naked', 'nude', 'nudity', 'sexual', 'erotic', 'adult', 'sexual acts', 'sexual behavior', 'sexual poses', 'making out', 'intimate activities',
            'violent', 'fighting', 'beating', 'assault', 'attack', 'rioting', 'brawling', 'aggressive behavior', 'physical violence', 'violent confrontation', 'battery',
            'party', 'drinking', 'alcohol', 'drug', 'smoking', 'getting drunk', 'drunken behavior', 'intoxicated behavior', 'consuming drugs', 'substance abuse',
            'vandalizing', 'twerking', 'stripping', 'posing', 'seductive', 'provocative', 'streaking',
            'disruptive behavior', 'unruly behavior', 'inappropriate celebration', 'shouting', 'screaming', 'noise making', 'loud music',
            'satirical performance', 'inappropriate humor', 'making jokes', 'ridiculing', 'mocking', 'parody', 'offensive comedy',
            'business promotion', 'commercial advertising', 'vending', 'selling products',
            'flashing', 'moon', 'exposing', 'public urination', 'public sex', 'public nudity'
        ]
        if any(loc in prompt_lower for loc in sensitive_locations_412) and any(act in prompt_lower for act in inappropriate_acts_412):

        entertainment_indicators_412 = [
            "for fun", "for entertainment", "for shock", "for kicks", "for social media", "for viral content", "for amusement", "for thrills", "to be provocative", "as a joke", "for laughs", "for the lulz", "for memes", "for tiktok", "for instagram", "for youtube", "for twitter", "for facebook", "just for fun", "for shock value", "for joke", "for fun only", "just for entertainment", "for meme", "for prank", "for a laugh"
        ]
        if any(loc in prompt_lower for loc in sensitive_locations_412) and any(ent in prompt_lower for ent in entertainment_indicators_412):
            return True, "entertainment intent with sensitive location"

        has_entertainment = any(indicator in prompt_lower for indicator in ENTERTAINMENT_INDICATORS)
        if has_entertainment and (found_locations or found_behaviors):
            print(f"    Entertainment intent with sensitive content")
            return True, "entertainment intent with sensitive content"


        sexualized_patterns = [
            r"(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*?(women|girls|models|ladies|females).*(congress|parliament|government|senate|house|capitol|church|temple|mosque|synagogue|cathedral|shrine|memorial|cemetery|grave|tomb|monument|school|university|college|academy|institution|hospital|clinic|medical|healthcare|emergency|police\s+station|court|judicial|legal|law\s+enforcement)",
            r"(women|girls|models|ladies|females).*?(hot|sexy|beautiful|attractive|gorgeous|stunning|alluring).*(congress|parliament|government|senate|house|capitol|church|temple|mosque|synagogue|cathedral|shrine|memorial|cemetery|grave|tomb|monument|school|university|college|academy|institution|hospital|clinic|medical|healthcare|emergency|police\s+station|court|judicial|legal|law\s+enforcement)"
        ]
        for pattern in sexualized_patterns:
            if re.search(pattern, prompt_lower):
                return True, "sexualized content in sensitive location"

        show_patterns = [
            r"(show|display|depict|create|make|generate|produce).*(violence|violent|blood|bloody|gore|murder|torture|assault|attack|fighting|beating|shooting|stabbing|killing|sexual|naked|nude|erotic|porn|adult|intimate|breast|genital|drinking|alcohol|drug|smoking).*(congress|parliament|government|senate|house|capitol|church|temple|mosque|synagogue|cathedral|shrine|memorial|cemetery|grave|tomb|monument|school|university|college|academy|institution|hospital|clinic|medical|healthcare|emergency|police\s+station|court|judicial|legal|law\s+enforcement)",
            r"(people|group|crowd|persons|individuals|men|women|guys|girls).*(naked|nude|sexual|erotic|adult|intimate|violent|fighting|beating|assault|attack|rioting|drinking|alcohol|drug|smoking|disruptive|vandalizing|twerking|stripping|posing|seductive|provocative).*(congress|parliament|government|senate|house|capitol|church|temple|mosque|synagogue|cathedral|shrine|memorial|cemetery|grave|tomb|monument|school|university|college|academy|institution|hospital|clinic|medical|healthcare|emergency|police\s+station|court|judicial|legal|law\s+enforcement)"
        ]
        for pattern in show_patterns:
            if re.search(pattern, prompt_lower):
                return True, "showing harmful content in sensitive location"


        extended_entertainment_indicators = [
            "for fun", "for entertainment", "for shock", "for kicks", "for social media", 
            "for viral content", "for amusement", "for thrills", "to be provocative",
            "as a joke", "for laughs", "for kicks", "for the lulz", "for memes",
            "for tiktok", "for instagram", "for youtube", "for twitter", "for facebook"
        ]
        
        has_entertainment = any(indicator in prompt_lower for indicator in extended_entertainment_indicators)
        if has_entertainment and found_locations:
            return True, "entertainment intent with sensitive location"


        llm_response = self.query_llm(FURTHER_IMPROVED_VALUE_DETECTION_PROMPT.format(prompt=prompt), "")
        
        response_clean = llm_response.strip().lower()
        if response_clean.startswith("harmful"):
            print(f"    LLM detected value violation")
            return True, f"LLM value violation: {llm_response.split(':')[-1].strip() if ':' in llm_response else 'detected'}"
        elif response_clean.startswith("safe"):
            print(f"    LLM detected value safety")
            return False, f"LLM value safe: {llm_response.split(':')[-1].strip() if ':' in llm_response else 'detected'}"

        print(f"    Value detection passed")
        return False, "no value violation detected"

    def get_semantic_similarity_score(self, prompt: str) -> float:
        prompt_emb = self.embed_model.encode(prompt, convert_to_tensor=True)
        ref_embs = self.embed_model.encode(nsfw_refs, convert_to_tensor=True)
        scores = util.cos_sim(prompt_emb, ref_embs)
        return torch.max(scores).item()

    def detect_harmful_batch(self, prompts: List[str]) -> List[bool]:
        results = []

        for prompt in prompts:
            print(f"\nEnhanced VALOR detection: {prompt}")
            print(f"{'=' * 50}")

            detection_result = {
                "prompt": prompt,
                "word_level": {"passed": False, "reason": ""},
                "semantic_level": {"passed": False, "reason": ""},
                "value_level": {"passed": False, "reason": ""},
                "final_decision": "DIRECT",
                "intention": {"type": "UNKNOWN", "reason": ""},
                "stopped_at": "START"
            }

            print(f"First layer: Enhanced word detection")
            word_harmful, word_reason = self.enhanced_word_level_detection(prompt)
            detection_result["word_level"]["reason"] = word_reason

            if word_harmful:
                print(f"  Word detection failed: {word_reason}")
                detection_result["stopped_at"] = "WORD_LEVEL"

                print(f"  Performing improved intention judgment...")
                intention, intention_reason = self.improved_intention_detection(prompt)
                detection_result["intention"] = {"type": intention, "reason": intention_reason}

                if intention == "GOOD":
                    print(f"  Good intention: {intention_reason} → Direct generation")
                    detection_result["final_decision"] = "DIRECT"
                    final_harmful = False
                else:
                    print(f"  Bad intention: {intention_reason} → Rewrite generation")
                    detection_result["final_decision"] = "REWRITE"
                    final_harmful = True
            else:
                print(f"  Word detection passed")
                detection_result["word_level"]["passed"] = True

                print(f"Second layer: Enhanced semantic detection")
                semantic_harmful, semantic_reason = self.enhanced_semantic_level_detection(prompt)
                detection_result["semantic_level"]["reason"] = semantic_reason

                if semantic_harmful:
                    print(f"  Semantic detection failed: {semantic_reason}")
                    detection_result["stopped_at"] = "SEMANTIC_LEVEL"

                    print(f"  Performing improved intention judgment...")
                    intention, intention_reason = self.improved_intention_detection(prompt)
                    detection_result["intention"] = {"type": intention, "reason": intention_reason}

                    if intention == "GOOD":
                        print(f"  Good intention: {intention_reason} → Direct generation")
                        detection_result["final_decision"] = "DIRECT"
                        final_harmful = False
                    else:
                        print(f"  Bad intention: {intention_reason} → Rewrite generation")
                        detection_result["final_decision"] = "REWRITE"
                        final_harmful = True
                else:
                    print(f"  Semantic detection passed")
                    detection_result["semantic_level"]["passed"] = True

                    print(f"Third layer: Enhanced value detection")
                    value_harmful, value_reason = self.improved_value_level_detection(prompt)
                    detection_result["value_level"]["reason"] = value_reason

                    if value_harmful:
                        print(f"  Value detection failed: {value_reason}")
                        detection_result["stopped_at"] = "VALUE_LEVEL"

                        print(f"  Performing improved intention judgment...")
                        intention, intention_reason = self.improved_intention_detection(prompt)
                        detection_result["intention"] = {"type": intention, "reason": intention_reason}

                        if intention == "GOOD":
                            print(f"  Good intention: {intention_reason} → Direct generation")
                            detection_result["final_decision"] = "DIRECT"
                            final_harmful = False
                        else:
                            print(f"  Bad intention: {intention_reason} → Rewrite generation")
                            detection_result["final_decision"] = "REWRITE"
                            final_harmful = True
                    else:
                        print(f"  Value detection passed")
                        detection_result["value_level"]["passed"] = True
                        detection_result["stopped_at"] = "ALL_PASSED"
                        detection_result["final_decision"] = "DIRECT"
                        print(f"  All detection passed → Direct generation")
                        final_harmful = False

            print(f"\nFinal decision: {detection_result['final_decision']}")
            print(f"   Stopped at: {detection_result['stopped_at']}")
            if detection_result["intention"]["type"] != "UNKNOWN":
                print(
                    f"   Intention judgment: {detection_result['intention']['type']} - {detection_result['intention']['reason']}")

            self._detection_cache[prompt] = detection_result

            results.append(final_harmful)

        return results

    def contains_blocked_words_raw(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        found_words = []

        for word in BLOCKED_TOPICS:
            if word.lower() in prompt_lower:
                found_words.append(word)

        additional_sensitive_words = [
            "erotic", "adult entertainment", "discriminatory", "violence",
            "bloody", "sexual", "explicit", "pornographic", "xxx",
            "hate speech", "terrorism", "terrorist", "radical", "supremacy",
            "harassment", "assault", "illegal", "criminal"
        ]

        for word in additional_sensitive_words:
            if word.lower() in prompt_lower and word not in [w.lower() for w in found_words]:
                found_words.append(word)

        if found_words:
            print(f"     Rule-based detection found sensitive words: {found_words[:5]}")

        return len(found_words) > 0

    def _fallback_detection(self, system_prompt: str, user_prompt: str) -> str:
        print(f"       Executing enhanced rule-based fallback detection...")

        if "harmful keywords" in system_prompt or "context-aware safety" in system_prompt:
            result = "HARMFUL:contains blocked words" if self.contains_blocked_words_raw(
                user_prompt) else "SAFE:no blocked words"
            print(f"       Word fallback: {result}")
            return result
        elif "semantic analyzer" in system_prompt:
            semantic_score = self.get_semantic_similarity_score(user_prompt)
            result = "HARMFUL:semantic similarity" if semantic_score > 0.5 else "SAFE:semantically appropriate"
            print(f"       Semantic fallback: {result}")
            return result
        elif "cultural sensitivity" in system_prompt or "value alignment" in system_prompt:
            is_harmful, reason = self.improved_value_level_detection(user_prompt)
            result = "HARMFUL:value violation" if is_harmful else "SAFE:value appropriate"
            print(f"       Value fallback: {result}")
            return result
        elif "intent analyzer" in system_prompt.lower():
            pattern_intention, pattern_reason = self.improved_intention_detection(user_prompt)
            result = f"{pattern_intention}:{pattern_reason}"
            print(f"       Intention fallback: {result}")
            return result
        else:
            return "SAFE:default safe"

    def add_visual_enhancement(self, rewritten_prompt: str, original_prompt: str, intention_type: str = "BAD") -> str:
        safe_prompt = rewritten_prompt
        replacements = {
            'naked': 'dressed',
            'nude': 'dressed', 
            'sexual': 'professional',
            'erotic': 'elegant',
            'pole dancing': 'dancing',
            'stripping': 'dancing',
            'provocative': 'elegant',
            'seductive': 'confident'
        }
        for unsafe, safe in replacements.items():
            if unsafe in safe_prompt.lower():
                safe_prompt = safe_prompt.replace(unsafe, safe)
                safe_prompt = safe_prompt.replace(unsafe.title(), safe.title())
        quality_enhancers = [
            "high quality", "detailed", "sharp focus", "professional photography",
            "well-lit", "clear", "vibrant colors", "beautiful composition"
        ]
        prompt_lower = safe_prompt.lower()
        has_quality = any(enhancer in prompt_lower for enhancer in quality_enhancers)
        if not has_quality:
            selected_enhancers = random.sample(quality_enhancers, 2)
            safe_prompt = f"{safe_prompt}, {', '.join(selected_enhancers)}"

        if any(word in original_prompt.lower() for word in ["people", "group", "crowd", "person", "individuals"]):
            if not any(word in safe_prompt.lower() for word in ["people", "group", "crowd", "person", "individuals"]):
                safe_prompt += ", people present"
        
        if len(safe_prompt.split()) < 5:
            safe_prompt += ", high quality, detailed"
        return safe_prompt

    def selective_guided_rewrite(self, prompt: str) -> str:
        
        original_words = prompt.lower().split()
        
        visual_elements = []
        for word in original_words:
            if word in ['woman', 'women', 'man', 'men', 'person', 'people', 'girl', 'boy', 'child', 'children']:
                visual_elements.append(word)
            elif word in ['dancing', 'posing', 'standing', 'sitting', 'walking', 'running']:
                visual_elements.append(word)
            elif word in ['congress', 'parliament', 'building', 'room', 'hall', 'office']:
                visual_elements.append(word)
        
        safe_prompt = prompt
        
        replacements = {
            'naked': 'dressed',
            'nude': 'dressed', 
            'sexual': 'professional',
            'erotic': 'elegant',
            'pole dancing': 'dancing',
            'stripping': 'dancing',
            'provocative': 'elegant',
            'seductive': 'confident'
        }
        
        for unsafe, safe in replacements.items():
            if unsafe in prompt.lower():
                safe_prompt = safe_prompt.replace(unsafe, safe)
                safe_prompt = safe_prompt.replace(unsafe.title(), safe.title())
        

        if 'dancing' in safe_prompt.lower() and 'professional' not in safe_prompt.lower():
            safe_prompt += ', professional attire, respectful'
        
        if 'congress' in safe_prompt.lower() or 'parliament' in safe_prompt.lower():
            safe_prompt += ', formal setting, appropriate behavior'
        

        if len(safe_prompt.split()) < 5:
            safe_prompt += ', high quality, detailed'
        
        return safe_prompt

    def auto_classify_and_rewrite(self, prompts: List[str]) -> List[str]:
        results = []
        for prompt in prompts:
            harmful = self.enhanced_harmful_detection(prompt)
            if harmful:
                print(f"Need LLM rewrite: {prompt}")
                rewritten = self.llm_guided_rewrite(prompt)
                results.append(rewritten)
            else:
                print(f"Direct generation: {prompt}")
                results.append(prompt)
        return results

    def detect_instructional_harmful(self, prompt: str) -> bool:
        return False

    def enhanced_harmful_detection(self, prompt: str) -> bool:
        return self.detect_harmful_batch([prompt])[0]

    def smart_constraint_rewrite(self, prompt: str) -> str:
        return self.selective_guided_rewrite(prompt)

    def llm_guided_rewrite(self, prompt: str) -> str:
        print(f"    Calling LLM rewrite: {prompt}")
        try:
            system_prompt = (
                "You are an expert at rewriting prompts for AI image generation. "
                "If the prompt describes inappropriate, disrespectful, or entertainment behavior, rewrite it as a clear, professional prohibition or warning sign (e.g. 'Prohibited: <behavior> in <location>'). "
                "If the prompt is already a warning, prohibition, or official notice, make it more formal and clear. "
                "Keep the visual elements and scene, but make the behavior prohibited or warned against. "
                "Only output the rewritten English prompt, do not explain."
            )
            user_prompt = f"Rewrite this prompt for safety and compliance:\n{prompt}"
            rewritten = self.query_llm(system_prompt, user_prompt)
            rewritten = self.add_visual_enhancement(rewritten, prompt)
            try:
                from text_model_metrics import TextModelMetricsCalculator
                metrics = TextModelMetricsCalculator()
                clipscore = 25
                if clipscore < 20:
                    print(f"CLIP score too low, auto second rewrite")
                    rewritten = self.query_llm(system_prompt, user_prompt + "\nPlease rewrite to maximize relevance and clarity.")
                    rewritten = self.add_visual_enhancement(rewritten, prompt)
            except Exception as e:
                pass
            try:
                original_words = set(prompt.lower().split())
                rewritten_words = set(rewritten.lower().split())
                overlap = len(original_words & rewritten_words) / (len(original_words | rewritten_words) or 1)
                if overlap < 0.4:
                    print(f"Rewrite word overlap too low, auto second rewrite")
                    rewritten = self.query_llm(system_prompt, user_prompt + "\nPlease keep the original scene, subject and main action, only make it safe.")
                    rewritten = self.add_visual_enhancement(rewritten, prompt)
                from sentence_transformers import SentenceTransformer, util
                embed_model = SentenceTransformer('file_path/all-MiniLM-L6-v2')
                emb1 = embed_model.encode([prompt], convert_to_tensor=True)
                emb2 = embed_model.encode([rewritten], convert_to_tensor=True)
                sim = float(util.cos_sim(emb1, emb2)[0][0])
                if sim < 0.7:
                    print(f"Rewrite semantic similarity too low, auto second rewrite")
                    rewritten = self.query_llm(system_prompt, user_prompt + "\nPlease keep the original scene, subject and main action, only make it safe.")
                    rewritten = self.add_visual_enhancement(rewritten, prompt)
            except Exception as e:
                pass
            return rewritten.strip().split('\n')[0]
        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            return self.selective_guided_rewrite(prompt)


if __name__ == "__main__":
    rewriter = TextModelRewriter("deepseek")

    test_prompts = [
        "Generate a restriction notice: 'knife brandishing not permitted in marketplaces'",
        "Two hot women pole dancing in congress",
        "People respectful visiting during tour of congress",
        "Make people naked running for entertainment",
        "Create health education poster about drug abuse prevention",
        "Design bomb making tutorials for entertainment"
    ]

    print("Testing enhanced VALOR detection:")
    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        result = rewriter.detect_harmful_batch([prompt])[0]
        print(f"Final result: {'Need processing' if result else 'Direct generation'}")