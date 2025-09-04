# omy_f3m_llm_parser.py
# build: 2025-08-13-LLM-ONLY

import re
import json
import time
from typing import Dict, Any, Optional, List, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)


# --------------------
# Settings
# --------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 200

# CUDA / dtype
USE_CUDA = True
PREFERRED_DTYPE = "fp16"   # "fp16" or "bf16"

# Outlines (structured decoding, optional)
USE_OUTLINES = False

# --------------------
# Outlines (optional)
# --------------------
try:
    from outlines import generate
    from outlines.models.transformers import Transformers as OutlinesTFModel
    HAS_OUTLINES = True
except Exception:
    HAS_OUTLINES = False

# --------------------
# Single-load guard
# --------------------
_MODEL = None
_TOKENIZER = None
_OUTLINES_LM = None

def _load_model_once():
    global _MODEL, _TOKENIZER, _DEVICE, _OUTLINES_LM
    if _MODEL is not None and _TOKENIZER is not None:
        return

    # --- Device / dtype ---
    has_cuda = torch.cuda.is_available()
    _DEVICE = "cuda" if (USE_CUDA and has_cuda) else "cpu"
    compute_dtype = None
    if _DEVICE == "cuda":
        compute_dtype = torch.float16 if PREFERRED_DTYPE.lower() == "fp16" else torch.bfloat16

    print(f"➡️  Device: {_DEVICE}")
    if _DEVICE == "cuda":
        try:
            print(f"🎯 CUDA device: {torch.cuda.get_device_name(0)} | dtype={compute_dtype}")
        except Exception:
            pass

    # --- Tokenizer / Model ---
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto" if _DEVICE == "cuda" else None,
        torch_dtype=compute_dtype if _DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if _DEVICE == "cuda" and not hasattr(_MODEL, "hf_device_map"):
        _MODEL.to(dtype=compute_dtype)
        _MODEL.to("cuda")

    # Deterministic decoding
    try:
        _MODEL.generation_config = GenerationConfig.from_model_config(_MODEL.config)
    except Exception:
        _MODEL.generation_config = GenerationConfig()
    _MODEL.generation_config.do_sample = False
    _MODEL.generation_config.temperature = None
    _MODEL.generation_config.top_p = None
    _MODEL.generation_config.top_k = None
    _MODEL.generation_config.num_beams = 1

    # --- Outlines wrapper (optional) ---
    _OUTLINES_LM = None
    if HAS_OUTLINES and USE_OUTLINES:
        try:
            _OUTLINES_LM = OutlinesTFModel(MODEL_ID, device=_DEVICE)
            print("✅ Outlines ready (structured decoding)")
        except Exception as e:
            _OUTLINES_LM = None
            print(f"⚠️ Outlines init failed → vanilla only: {e}")

_load_model_once()
model = _MODEL
tokenizer = _TOKENIZER
device = _DEVICE
outlines_lm = _OUTLINES_LM

# --------------------
# Schema & Keys
# --------------------
SCHEMA_KEYS = ["action", "direction", "value", "unit", "xyz"]

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action":    {"type": ["string", "null"]},
        "direction": {"type": ["string", "null"]},
        "value":     {"type": ["number", "string", "null"]},  # allow "keep"
        "unit":      {"type": ["string", "null"]},
        "xyz": {
            "anyOf": [
                {"type": "null"},
                {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
            ]
        }
    },
    "required": ["action", "direction", "value", "unit", "xyz"],
    "additionalProperties": False
}

def _empty_schema() -> Dict[str, Any]:
    return {k: None for k in SCHEMA_KEYS}

# --------------------
# Prompt (Chat template)
# --------------------
SYSTEM_HINT = """You are a precise JSON command formatter for robot control.
Respond ONLY with one JSON object: {"action":..., "direction":..., "value":..., "unit":..., "xyz":...}
Do not add explanations.

RULES:
- Copy numbers exactly.
- If input has exactly 3 numbers (e.g. "go to x y z"), use action="move_xyz" and xyz=[x,y,z].
- Angle units → action="rotate".
- Distance units → action="move".
- Directions: left/right/up/down only.
- "turn"/"rotate"/"spin" → action="rotate".
- "gripper open/close/reset" → action="gripper".
- "home"/"initial pose"/"reset pose" → action="initialize".

CONTINUOUS/STOP:
- If the text indicates continuous motion, such as "keep", "keep going", "until stop", or "when I say stop",
  then set value="keep" (a string), unit=null. Do NOT invent a numeric value.
- If the user says exactly "stop" (or a clear stop intent), return:
  {"action":"stop","direction":null,"value":null,"unit":null,"xyz":null}.

DEFAULTS (rotate only):
- If the input is a rotate command with a direction (left/right/up/down) but has no numeric amount
  and is not a continuous command ("keep"), set value=10 and unit="degree".
- Do NOT apply defaults to gripper/move/xyz or to any non-rotate action.
"""

FEW_SHOTS = [
    # ROTATE with explicit values
    ("turn right 45 degree", {"action":"rotate","direction":"right","value":45,"unit":"degree","xyz":None}),
    ("turn left 30 degree",  {"action":"rotate","direction":"left","value":30,"unit":"degree","xyz":None}),
    ("rotate clockwise 90 degree", {"action":"rotate","direction":"right","value":90,"unit":"degree","xyz":None}),
    ("rotate counterclockwise 60 degree", {"action":"rotate","direction":"left","value":60,"unit":"degree","xyz":None}),

    # ROTATE with no value → defaults kick in (10 degree)
    ("turn right", {"action":"rotate","direction":"right","value":10,"unit":"degree","xyz":None}),
    ("turn left",  {"action":"rotate","direction":"left","value":10,"unit":"degree","xyz":None}),
    ("rotate up",  {"action":"rotate","direction":"up","value":10,"unit":"degree","xyz":None}),
    ("rotate down",{"action":"rotate","direction":"down","value":10,"unit":"degree","xyz":None}),

    # MOVE
    ("move up 10 cm",        {"action":"move","direction":"up","value":10,"unit":"cm","xyz":None}),
    ("move down 5 cm",       {"action":"move","direction":"down","value":5,"unit":"cm","xyz":None}),
    ("move right 10 inch",   {"action":"move","direction":"right","value":10,"unit":"inch","xyz":None}),
    ("go to right 10 mm",    {"action":"move","direction":"right","value":10,"unit":"mm","xyz":None}),

    # ROTATE numeric up/down
    ("rotate up 10 degree",   {"action":"rotate","direction":"up","value":10,"unit":"degree","xyz":None}),
    ("rotate down 15 degree", {"action":"rotate","direction":"down","value":15,"unit":"degree","xyz":None}),

    # CONTINUOUS
    ("turn right keep", {"action":"rotate","direction":"right","value":"keep","unit":None,"xyz":None}),
    ("turn right keep when i say stop", {"action":"rotate","direction":"right","value":"keep","unit":None,"xyz":None}),

    # STOP
    ("stop", {"action":"stop","direction":None,"value":None,"unit":None,"xyz":None}),

    # GRIPPER / INITIALIZE / XYZ
    ("gripper open",             {"action":"gripper","direction":"open","value":None,"unit":None,"xyz":None}),
    ("please gripper close",     {"action":"gripper","direction":"close","value":None,"unit":None,"xyz":None}),
    ("reset gripper",            {"action":"gripper","direction":"reset","value":None,"unit":None,"xyz":None}),
    ("go to initial pose",       {"action":"initialize","direction":None,"value":None,"unit":None,"xyz":None}),
    ("reset position",           {"action":"initialize","direction":None,"value":None,"unit":None,"xyz":None}),
    ("go to 0.3 0.2 0.1",        {"action":"move_xyz","direction":None,"value":None,"unit":None,"xyz":[0.3,0.2,0.1]}),
]

def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]

def build_chat_messages(user_text: str):
    numbers = _extract_numbers(user_text)
    shots = []
    for inp, out in FEW_SHOTS:
        shots.append({"role": "user", "content": f"Input: {inp}\nOutput:"})
        shots.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    return (
        [{"role": "system", "content": SYSTEM_HINT.strip()},
         {"role": "system", "content": f"INPUT NUMBERS: {json.dumps(numbers)}"}]
        + shots
        + [{"role": "user", "content": f"Input: {user_text}\nOutput:"}]
    )

def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------------------
# Stopping criteria
# --------------------
class StopOnPatterns(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer: AutoTokenizer, start_len: int):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.start_len = start_len
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.start_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        for s in self.stop_strings:
            if s in text:
                return True
        return False

def _extract_first_json_after_marker(full_text: str, marker: str = "\nOutput:") -> Optional[str]:
    idx = full_text.rfind(marker)
    if idx == -1:
        return None
    segment = full_text[idx + len(marker):]
    start = segment.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(segment)):
        ch = segment[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return segment[start:i+1].strip()
    return None

# --------------------
# Decoding
# --------------------
def outlines_decode_json(messages: Union[str, List[Dict[str, str]]]) -> Optional[Dict[str, Any]]:
    if not (HAS_OUTLINES and USE_OUTLINES and outlines_lm is not None):
        return None
    try:
        text = messages if isinstance(messages, str) else _messages_to_text(messages)
        json_gen = generate.json(JSON_SCHEMA)
        out_text = json_gen(outlines_lm, text, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
        jtxt = _extract_first_json_after_marker(out_text, marker="\nOutput:")
        if jtxt:
            return json.loads(jtxt)
        return json.loads(out_text)
    except Exception:
        return None

def vanilla_decode_json(messages: Union[str, List[Dict[str, str]]]) -> Optional[Dict[str, Any]]:
    text = messages if isinstance(messages, str) else _messages_to_text(messages)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_len = inputs["input_ids"].shape[1]
    stop_criteria = StoppingCriteriaList([
        StopOnPatterns(
            stop_strings=["\nInput:", "\nAssistant:", "\nUser:"],
            tokenizer=tokenizer,
            start_len=start_len,
        )
    ])
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stop_criteria,
        )
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    jtxt = _extract_first_json_after_marker(decoded, marker="\nOutput:")
    if not jtxt:
        m = re.search(r"\{[\s\S]*?\}", decoded)
        if not m:
            return None
        jtxt = m.group(0)
    try:
        return json.loads(jtxt)
    except Exception:
        return None

# --------------------
# Defaults post-processing
# --------------------
def _apply_rotate_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """If rotate+direction and no numeric value and not 'keep', set 10 degree."""
    try:
        if not isinstance(data, dict):
            return data
        if data.get("action") != "rotate":
            return data
        direction = (data.get("direction") or "").lower()
        if direction not in ("left", "right", "up", "down"):
            return data
        val = data.get("value")
        # don't override continuous mode
        if isinstance(val, str) and val.strip().lower() == "keep":
            return data
        # treat empty string as missing
        if val is None or (isinstance(val, str) and val.strip() == ""):
            data["value"] = 10
            data["unit"] = "degree"
        return data
    except Exception:
        return data

# --------------------
# Public API
# --------------------
def parse_to_json(text: str) -> Dict[str, Any]:
    start = time.time()
    messages = build_chat_messages(text)

    # Fast-path: plain "stop" (lowercase/trim)
    if text.strip().lower() == "stop":
        data = {"action":"stop","direction":None,"value":None,"unit":None,"xyz":None}
        print(f"✅ Parsed in {time.time()-start:.2f}s (fast stop)")
        return data

    data = outlines_decode_json(messages)
    if data is None:
        data = vanilla_decode_json(messages)
    if data is None:
        print(f"⚠️ No JSON parsed in {time.time()-start:.2f}s")
        return _empty_schema()

    # apply rotate defaults (10 degree) only when appropriate
    data = _apply_rotate_defaults(data)

    print(f"✅ Parsed in {time.time()-start:.2f}s")
    return data

# --------------------
# Local test
# --------------------
if __name__ == "__main__":
    tests = [
        "turn right 45 degree",          # explicit
        "turn left",                     # default 10 deg
        "rotate up",                     # default 10 deg
        "rotate down keep",              # keep (no default)
        "go to 0.3 0.2 0.1",
        "gripper open",
        "turn right keep when i say stop",
        "stop",
    ]
    for t in tests:
        print(t, "->", json.dumps(parse_to_json(t), ensure_ascii=False))