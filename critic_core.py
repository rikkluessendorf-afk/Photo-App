"""Photo Critic core: rubric, scoring, Gemini calls, image prep.

No UI code lives here. Every photo is judged in its own stateless request
(rubric + that one image), so photo #500 gets the same standards as photo #1
and the critic can't drift into cheerleading over a long session.
"""

import csv
import io
import json
import os
import random
import threading
import time
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel

try:  # iPhone HEIC support (optional)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

ANCHOR_DIR = Path(__file__).parent / "anchors"


def cfg(name: str, default, cast=str):
    """Read a setting from the environment (the app copies Streamlit secrets there)."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return cast(raw)
    except Exception:
        return default


def max_batch() -> int:
    return cfg("MAX_BATCH", 10, int)


# --------------------------------------------------------------------------
# Rubric: categories, weights, tiers  (edit here to retune)
# --------------------------------------------------------------------------
# key, display label, short label, weight.  Weights must sum to 1.0.
CATEGORIES = [
    ("light", "Use of Light and Shadow", "Light", 0.10),
    ("color", "Use of Color or B&W", "Color", 0.10),
    ("composition", "Composition", "Comp", 0.10),
    ("technical", "Technical", "Tech", 0.10),
    ("subject", "Choice of Subject", "Subject", 0.10),
    ("execution", "Overall Execution", "Exec", 0.50),
]
assert abs(sum(c[3] for c in CATEGORIES) - 1.0) < 1e-9

# (minimum composite, tier label, short tier)
TIERS = [
    (85.0, "Tier S", "S"),
    (75.0, "Tier A", "A"),
    (65.0, "Tier B", "B"),
    (0.0, "The Field", "Field"),
]


def tier_for(composite: float):
    for floor, label, short in TIERS:
        if composite >= floor:
            return label, short
    return TIERS[-1][1], TIERS[-1][2]


def compute_composite(scores: dict) -> float:
    total = sum(scores[key] * weight for key, _, _, weight in CATEGORIES)
    return round(total, 1)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a veteran photography instructor and gallery-level photo editor. A dedicated student has asked for an honest critique. Your tone is that of a good teacher: warm, direct, specific, and invested in their growth. You praise what truly works, name what does not, and show how to repair it. Your NUMBERS are rigorous and unsentimental; your WORDS are kind. Empty praise is a disservice to the student.

SCORING SCALE (1-100, used for every category)
  1-10   Catastrophic. Accidental, unusable, no discernible intent.
  11-25  Very poor. Fundamental failures in most respects.
  26-40  Weak snapshot. Obvious problems; little evidence of decisions.
  41-55  Ordinary snapshot. Where most casual phone photos belong.
  56-64  Good snapshot / developing hobbyist. Some real decisions, still clear flaws.
  65-74  Very well done amateur. Intentional, controlled, clearly practiced hobbyist work. (65 is where this level starts.)
  75-84  Professional, competent AND artistic. Could sell for commercial or practical reasons: places, people, products, editorial, stock. (75 is where this level starts.)
  85-94  Fine art. Distinctive vision and mastery; could sell as art on its own standing. (85 is where this level starts.)
  95-100 Once-in-a-generation masterwork. Essentially never awarded; above 95 requires an extraordinary reason.

CALIBRATION RULES
- Be stingy upward. A score of 65+ needs specific visible evidence of skill and intent. A merely pleasant or technically clean photo is NOT 65.
- A pretty subject is not a good photograph. Conventional sunsets, flowers, pets, landmarks, and "nice view" shots usually sit in the 40s-50s unless something distinguishes them.
- Technical cleanliness alone cannot lift Overall Execution much above the mid-60s without vision, intent, and impact.
- Use the whole range and ordinary numbers like 43, 58, 67, 71. Avoid clustering at round numbers.
- Judge the photograph itself, not the effort, the story, or what the student says they hoped for.
- Judge by the standards of the photo's genre (street, portrait, landscape, macro, architecture, wildlife, etc.). Deliberate blur, grain, or crushed blacks are fine when they serve the image; accidental ones are not.
- If reference photos are supplied, they show approximate levels, not tier boundaries. Use them to calibrate your overall standard.
- Never inflate to be encouraging. If you notice yourself softening a number, lower it.

CATEGORIES
- Use of Light and Shadow: quality, direction, contrast, and mood of light; control of highlights and shadows; whether light is used with intent.
- Use of Color or B&W: palette, harmony and contrast, tonal range, restraint in grading, and whether the choice of color or monochrome serves the image.
- Composition: framing, balance, geometry, layering, leading lines, edges, negative space, cropping, and how the eye travels through the frame.
- Technical: focus, sharpness, exposure, noise, dynamic range, white balance, level horizon, distortion, processing artifacts, resolution, judged against the photo's intent.
- Choice of Subject: originality, interest, meaning, emotional pull, strength of the idea. This is NOT how pretty the subject is; it is whether the photo has something to say or just records something familiar.
- Overall Execution: the holistic verdict. How completely the finished photograph realizes a clear intent: impact, cohesion, polish, and whether a viewer would stop and stay. It is NOT an average of the other five, though it will usually sit within roughly 15 points of them unless you have a clear reason.

WRITING THE CRITIQUE
- For each category, write a one-to-two sentence "note" citing concrete evidence from THIS image (name actual elements: "the bright window at upper left pulls the eye away from her face"). Write the note BEFORE the score so the score follows from the evidence.
- "what_works": 2-4 sentences of earned praise, specific to this photo.
- "what_doesnt": honest, specific problems. Always name at least one concrete weakness, even for strong photos.
- "repairs": actionable fixes. Separate what can be repaired in editing or cropping from what requires reshooting.
- "closing": one or two sentences in a teacher's voice that invite discussion, such as asking about their intent or inviting disagreement. Do not flatter.
- Avoid hollow superlatives (stunning, breathtaking, amazing, masterpiece) unless the scores are 90+.
- "photo_type": a short genre label such as "Street portrait" or "Coastal landscape".
- Keep the whole critique roughly 250-400 words.
Respond only with JSON matching the requested schema."""

DISCUSSION_SYSTEM = """You are the photography instructor who just critiqued the student's photo (shown to you, along with your earlier critique and scores below). The student now wants to discuss it.

- Stay in a teacher's voice: warm, direct, specific, brief (usually under 200 words).
- Take pushback seriously. If the student makes a good point or reveals intent or context that changes your view, say so plainly and say which category would move and roughly by how many points, and why.
- If their argument does not hold up, hold your ground politely and explain what in the image supports your judgment. Do not cave to pressure, repetition, or flattery.
- Never invent details that are not visible in the photo. If you cannot tell something, say so.
- Answer practical questions (how to reshoot, how to edit, what to study) concretely.
- Revised scores you mention in conversation are for discussion only; they do not alter the saved results.

YOUR EARLIER CRITIQUE:
{critique}
"""


# --------------------------------------------------------------------------
# Structured output schema (field order = generation order: evidence, then score)
# --------------------------------------------------------------------------
class CategoryResult(BaseModel):
    note: str
    score: int


class Critique(BaseModel):
    photo_type: str
    light: CategoryResult
    color: CategoryResult
    composition: CategoryResult
    technical: CategoryResult
    subject: CategoryResult
    execution: CategoryResult
    what_works: str
    what_doesnt: str
    repairs: str
    closing: str


# --------------------------------------------------------------------------
# Gemini plumbing: client, throttle, retries
# --------------------------------------------------------------------------
class DailyQuotaError(Exception):
    pass


_client = None
_client_lock = threading.Lock()
_rl_lock = threading.Lock()
_last_call = 0.0


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set. Add it under the app's Secrets.")
            _client = genai.Client(api_key=key)
        return _client


def throttle():
    """Space calls out so the free tier's per-minute limit is respected."""
    global _last_call
    with _rl_lock:
        wait = cfg("MIN_SECONDS_BETWEEN_CALLS", 5.0, float) - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def generate_with_retry(**kwargs):
    delay = 10.0
    for attempt in range(5):
        throttle()
        try:
            return get_client().models.generate_content(**kwargs)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            code = getattr(e, "code", None)
            if "PerDay" in msg or "per day" in msg.lower():
                raise DailyQuotaError(
                    "The free daily quota for this model looks used up. It resets daily; "
                    "try again later, or set GEMINI_MODEL to a different Flash model."
                ) from e
            retryable = code in (429, 500, 502, 503, 504) or any(
                t in msg for t in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE")
            )
            if not retryable or attempt == 4:
                raise
            time.sleep(delay + random.uniform(0, 3))
            delay *= 1.8


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def prepare_image(src, max_edge: int = None) -> bytes:
    """Fix rotation, flatten to RGB, shrink, and return JPEG bytes. `src` is a path or file-like."""
    if max_edge is None:
        max_edge = cfg("MAX_IMAGE_EDGE", 2000, int)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


@lru_cache(maxsize=1)
def load_anchors():
    """Optional calibration photos: anchors/anchors.json + image files."""
    cfg_path = ANCHOR_DIR / "anchors.json"
    if not cfg_path.exists():
        return []
    try:
        items = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for item in items[: cfg("MAX_ANCHORS", 6, int)]:
        p = ANCHOR_DIR / str(item.get("file", ""))
        if p.exists() and "score" in item:
            try:
                out.append(
                    {
                        "bytes": prepare_image(p, 768),
                        "score": item["score"],
                        "note": item.get("note", ""),
                    }
                )
            except Exception:
                continue
    return out


def image_part(data: bytes):
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def clamp_score(v) -> int:
    try:
        return max(1, min(100, int(round(float(v)))))
    except Exception:
        return 1


def critique_text(r: dict) -> str:
    lines = [f"{r['tier_label']}, composite {r['composite']:.1f}/100 ({r['photo_type']})"]
    for key, label, _, weight in CATEGORIES:
        c = r["cats"][key]
        lines.append(f"- {label}: {c['score']} ({int(weight * 100)}% weight): {c['note']}")
    lines.append(f"What works: {r['what_works']}")
    lines.append(f"What doesn't: {r['what_doesnt']}")
    lines.append(f"Repairs: {r['repairs']}")
    lines.append(f"Closing: {r['closing']}")
    return "\n".join(lines)


def evaluate(name: str, jpeg: bytes) -> dict:
    parts = []
    anchors = load_anchors()
    if anchors:
        parts.append(
            types.Part.from_text(
                text="Reference photos follow, each with the instructor's approximate overall level. "
                "They are calibration points, not tier boundaries."
            )
        )
        for a in anchors:
            parts.append(
                types.Part.from_text(
                    text=f"Reference photo, approx. overall level {a['score']}/100. {a['note']}"
                )
            )
            parts.append(image_part(a["bytes"]))
    parts.append(
        types.Part.from_text(
            text=f"Now critique this photograph (filename: {name}). Apply the full rubric and scale."
        )
    )
    parts.append(image_part(jpeg))

    resp = generate_with_retry(
        model=cfg("GEMINI_MODEL", "gemini-flash-latest"),
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=cfg("TEMPERATURE", 0.2, float),
            response_mime_type="application/json",
            response_schema=Critique,
        ),
    )
    data = getattr(resp, "parsed", None)
    if not isinstance(data, Critique):
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("The model returned nothing (the image may have been blocked).")
        data = Critique.model_validate_json(text)

    cats, scores = {}, {}
    for key, _, _, _ in CATEGORIES:
        c = getattr(data, key)
        scores[key] = clamp_score(c.score)
        cats[key] = {"score": scores[key], "note": c.note.strip()}
    composite = compute_composite(scores)
    tier_label, tier_short = tier_for(composite)
    return {
        "name": name,
        "jpeg": jpeg,
        "photo_type": data.photo_type.strip(),
        "cats": cats,
        "composite": composite,
        "tier_label": tier_label,
        "tier_short": tier_short,
        "what_works": data.what_works.strip(),
        "what_doesnt": data.what_doesnt.strip(),
        "repairs": data.repairs.strip(),
        "closing": data.closing.strip(),
    }


def critique_upload(name: str, data: bytes) -> dict:
    """Prepare an uploaded photo's bytes and critique it."""
    return evaluate(name, prepare_image(io.BytesIO(data)))


def discussion_reply(r: dict, history: list, message: str) -> str:
    """Teacher's reply in the per-photo discussion. `history` is prior {'role','content'} turns."""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="This is the photograph we are discussing."),
                image_part(r["jpeg"]),
            ],
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text="Understood. I have the photo and my critique in front of me.")],
        ),
    ]
    for h in history:
        role = "model" if h["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=str(h["content"]))]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
    try:
        resp = generate_with_retry(
            model=cfg("GEMINI_MODEL", "gemini-flash-latest"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=DISCUSSION_SYSTEM.format(critique=critique_text(r)),
                temperature=0.4,
            ),
        )
        return (getattr(resp, "text", None) or "").strip() or "(No reply came back. Try rephrasing.)"
    except Exception as e:  # noqa: BLE001
        return f"(I couldn't reach the model just now: {str(e)[:200]})"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def csv_text(results) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        ["photo", "composite", "tier", "type"]
        + [c[1] for c in CATEGORIES]
        + ["what_works", "what_doesnt", "repairs", "closing"]
    )
    for r in sorted(results, key=lambda x: x["composite"], reverse=True):
        w.writerow(
            [r["name"], r["composite"], r["tier_label"], r["photo_type"]]
            + [r["cats"][k]["score"] for k, _, _, _ in CATEGORIES]
            + [r["what_works"], r["what_doesnt"], r["repairs"], r["closing"]]
        )
    return buf.getvalue()
