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

USE_CUDA = True
PREFERRED_DTYPE = "fp16"   # "fp16" or "bf16"

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

    try:
        _MODEL.generation_config = GenerationConfig.from_model_config(_MODEL.config)
    except Exception:
        _MODEL.generation_config = GenerationConfig()
    _MODEL.generation_config.do_sample = False
    _MODEL.generation_config.temperature = None
    _MODEL.generation_config.top_p = None
    _MODEL.generation_config.top_k = None
    _MODEL.generation_config.num_beams = 1

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
        "value":     {"type": ["number", "string", "null"]},
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
- Angle units → action="rotate" (except when the verb is 'look', then use action="look").
- Distance units → action="move".
- Directions: left/right/up/down/forward/backward only.
- Synonyms:
  * "back" → backward
  * "forwards" → forward
  * "look up" → {"action":"look","direction":"up"}
  * "look down" → {"action":"look","direction":"down"}
- "turn"/"rotate"/"spin" (any direction: left/right/up/down/forward/backward) → action="rotate".
- "go"/"move"/"walk"/"step" → action="move".
- "gripper open/close/reset" → action="gripper".
- "look" (without turn/rotate) -> action="look" (only up/down).
- "look left/right" → treat as rotate.
- "home"/"initial pose"/"reset pose" → action="initialize".

CONTINUOUS/STOP:
- If the text indicates continuous motion, such as "keep", "keep going", "until stop", or "when I say stop",
  then set value="keep" (a string), unit=null.
- If the user says exactly "stop", return:
  {"action":"stop","direction":null,"value":null,"unit":null,"xyz":null}.

DEFAULTS (rotate/look only):
- If the input is a rotate or look command with a direction (left/right/up/down/forward/backward)
  but has no numeric amount and is not a continuous command ("keep"), set value=10 and unit="degree".
"""

FEW_SHOTS = [
    # ROTATE with explicit values
    ("turn right 45 degree", {"action":"rotate","direction":"right","value":45,"unit":"degree","xyz":None}),
    ("turn left 30 degree",  {"action":"rotate","direction":"left","value":30,"unit":"degree","xyz":None}),
    ("turn up 15 degree",    {"action":"rotate","direction":"up","value":15,"unit":"degree","xyz":None}),
    ("turn down 20 degree",  {"action":"rotate","direction":"down","value":20,"unit":"degree","xyz":None}),

    # LOOK (only up/down)
    ("look up 20 degree",    {"action":"look","direction":"up","value":20,"unit":"degree","xyz":None}),
    ("look down 15 degree",  {"action":"look","direction":"down","value":15,"unit":"degree","xyz":None}),
    ("look up",              {"action":"look","direction":"up","value":10,"unit":"degree","xyz":None}),
    ("look down",            {"action":"look","direction":"down","value":10,"unit":"degree","xyz":None}),

    # ROTATE defaults (left/right) even if phrased as look
    ("look left",  {"action":"rotate","direction":"left","value":10,"unit":"degree","xyz":None}),
    ("look right", {"action":"rotate","direction":"right","value":10,"unit":"degree","xyz":None}),

    # MOVE forward/backward
    ("go forward 3 cm",   {"action":"move","direction":"forward","value":3,"unit":"cm","xyz":None}),
    ("go backward 5 cm",  {"action":"move","direction":"backward","value":5,"unit":"cm","xyz":None}),
    ("move back 2 cm",    {"action":"move","direction":"backward","value":2,"unit":"cm","xyz":None}),
    ("step forwards 10 cm",{"action":"move","direction":"forward","value":10,"unit":"cm","xyz":None}),

    # MOVE up/down/left/right
    ("move up 10 cm",        {"action":"move","direction":"up","value":10,"unit":"cm","xyz":None}),
    ("move down 5 cm",       {"action":"move","direction":"down","value":5,"unit":"cm","xyz":None}),
    ("move right 10 inch",   {"action":"move","direction":"right","value":10,"unit":"inch","xyz":None}),
    ("go to left 20 mm",     {"action":"move","direction":"left","value":20,"unit":"mm","xyz":None}),

    # CONTINUOUS
    ("turn right keep", {"action":"rotate","direction":"right","value":"keep","unit":None,"xyz":None}),
    ("look left keep until stop", {"action":"rotate","direction":"left","value":"keep","unit":None,"xyz":None}),

    # STOP
    ("stop", {"action":"stop","direction":None,"value":None,"unit":None,"xyz":None}),

    # GRIPPER / INITIALIZE / XYZ
    ("gripper open",         {"action":"gripper","direction":"open","value":None,"unit":None,"xyz":None}),
    ("gripper close",        {"action":"gripper","direction":"close","value":None,"unit":None,"xyz":None}),
    ("reset gripper",        {"action":"gripper","direction":"reset","value":None,"unit":None,"xyz":None}),
    ("go to initial pose",   {"action":"initialize","direction":None,"value":None,"unit":None,"xyz":None}),
    ("go to 0.3 0.2 0.1",    {"action":"move_xyz","direction":None,"value":None,"unit":None,"xyz":[0.3,0.2,0.1]}),
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
# Public API
# --------------------
def parse_to_json(text: str) -> Dict[str, Any]:
    start = time.time()
    messages = build_chat_messages(text)

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

    print(f"✅ Parsed in {time.time()-start:.2f}s")
    return data

# --------------------
# Local test
# --------------------
if __name__ == "__main__":
    tests = [
        "turn right 45 degree",
        "turn up 15 degree",
        "turn down 15 degree",
        "look up 15 degree",
        "look down",
        "look right",
        "look left",
        "go forward 3 cm",
        "go backward 5 cm",
        "move back 2 cm",
        "step forwards 10 cm",
        "gripper open",
        "stop",
    ]
    for t in tests:
        print(t, "->", json.dumps(parse_to_json(t), ensure_ascii=False))
