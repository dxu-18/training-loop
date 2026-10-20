"""Themed data pools so the same syntax can be asked about many real-world things.

A Theme is a set of named things (AI labs, pets, appliances, ASR models, ...)
plus numeric Measures that make sense for them (latency, weight, power draw, ...).
An Axis is a second dimension for 2-D data (days, epochs, color channels, ...).
Values are random but kept in plausible ranges; they are not real-world facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    name: str        # "power draw"
    unit: str        # "W" ("" for unitless)
    var: str         # suggested variable name, e.g. "power_w"
    lo: float
    hi: float
    decimals: int = 0   # 0 -> integers

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


@dataclass(frozen=True)
class Theme:
    key: str
    noun: str        # singular, e.g. "appliance"
    plural: str      # e.g. "appliances"
    items: tuple[str, ...]
    measures: tuple[Measure, ...]

    @property
    def var(self) -> str:
        """Variable name for the array of item names, e.g. "labs", "pets"."""
        return ident(self.plural.split()[-1])


@dataclass(frozen=True)
class Axis:
    noun: str
    plural: str
    labels: tuple[str, ...]

    @property
    def var(self) -> str:
        return ident(self.plural)


M = Measure


def _t(key, noun, plural, items, measures):
    return Theme(key, noun, plural, tuple(items), tuple(measures))


THEMES: tuple[Theme, ...] = (
    _t("ai_labs", "AI lab", "AI labs",
       ["OpenAI", "Anthropic", "Google DeepMind", "Meta FAIR", "Mistral AI", "Cohere", "xAI",
        "AI21 Labs", "Hugging Face", "Stability AI", "DeepSeek", "Qwen", "Moonshot AI", "Reka"],
       [M("median API latency", "ms", "latency_ms", 80, 900),
        M("monthly API requests", "millions", "requests_m", 5, 900),
        M("papers published this year", "", "papers", 3, 120),
        M("GPUs reserved", "", "gpus", 256, 32000)]),
    _t("pets", "pet", "pets",
       ["dog", "cat", "rabbit", "hamster", "parrot", "goldfish", "guinea pig", "ferret",
        "turtle", "canary", "gecko", "hedgehog", "chinchilla"],
       [M("weight", "kg", "weight_kg", 0.1, 35, 1),
        M("daily food", "g", "food_g", 5, 600),
        M("vet visits per year", "", "vet_visits", 0, 6),
        M("age", "years", "age_years", 1, 15)]),
    _t("appliances", "appliance", "appliances",
       ["refrigerator", "microwave", "dishwasher", "washing machine", "toaster", "air fryer",
        "kettle", "vacuum cleaner", "dryer", "blender", "coffee maker", "rice cooker", "oven",
        "air purifier"],
       [M("power draw", "W", "power_w", 20, 2500),
        M("daily use", "hours", "hours_per_day", 0.1, 24, 1),
        M("price", "$", "price_usd", 25, 1800),
        M("age", "years", "age_years", 0, 20)]),
    _t("asr_models", "speech-recognition model", "speech-recognition models",
       ["Whisper-large", "wav2vec 2.0", "Conformer", "HuBERT", "Parakeet", "Canary",
        "DeepSpeech", "SpeechT5", "Moonshine", "Kaldi TDNN", "QuartzNet", "Citrinet"],
       [M("word error rate", "%", "wer", 2, 35, 1),
        M("real-time factor", "", "rtf", 0.02, 1.5, 2),
        M("latency", "ms", "latency_ms", 40, 1200),
        M("parameters", "millions", "params_m", 20, 1600)]),
    _t("audio_clips", "audio clip", "audio clips",
       ["meeting", "lecture", "voicemail", "podcast", "interview", "call-center", "audiobook",
        "news", "standup", "sermon", "radio", "vlog"],
       [M("duration", "s", "duration_s", 2, 900, 1),
        M("signal-to-noise ratio", "dB", "snr_db", -5, 40, 1),
        M("speaking rate", "words/min", "wpm", 90, 210),
        M("silence ratio", "", "silence", 0, 0.8, 2)]),
    _t("vision_models", "vision model", "vision models",
       ["ResNet-50", "ViT-B/16", "EfficientNet-B0", "ConvNeXt-T", "YOLOv8n", "DETR",
        "CLIP ViT-L/14", "MobileNetV3", "Swin-T", "DINOv2", "RegNetY", "SAM"],
       [M("top-1 accuracy", "%", "acc", 55, 91, 1),
        M("inference time", "ms", "infer_ms", 2, 180, 1),
        M("parameters", "millions", "params_m", 3, 640),
        M("GFLOPs", "", "gflops", 0.2, 180, 1)]),
    _t("gen_models", "image/video generator", "image/video generators",
       ["Stable Diffusion XL", "FLUX.1", "Imagen", "Midjourney", "DALL-E 3", "Sora",
        "Runway Gen-3", "Pika", "Veo", "Kling", "SDXL-Turbo", "PixArt"],
       [M("generation time", "s", "gen_s", 0.5, 120, 1),
        M("FID", "", "fid", 4, 40, 1),
        M("CLIP score", "", "clip_score", 0.20, 0.36, 3),
        M("frames per second", "fps", "fps", 8, 60)]),
    _t("object_classes", "object class", "object classes",
       ["person", "car", "bicycle", "dog", "traffic light", "stop sign", "bus", "truck", "cat",
        "chair", "bottle", "laptop", "backpack"],
       [M("detections per frame", "", "dets", 0, 40),
        M("mean confidence", "", "conf", 0.3, 0.99, 2),
        M("mean box area", "px²", "area_px", 50, 90000)]),
    _t("datasets", "dataset", "datasets",
       ["MNIST", "CIFAR-10", "ImageNet", "COCO", "LibriSpeech", "Common Voice", "SQuAD", "GLUE",
        "WikiText-103", "LAION-400M", "The Pile", "C4", "FineWeb"],
       [M("size on disk", "GB", "size_gb", 0.05, 900, 2),
        M("samples", "thousands", "samples_k", 10, 90000),
        M("download time", "min", "download_min", 1, 600)]),
    _t("checkpoints", "training checkpoint", "training checkpoints",
       ["step-1k", "step-2k", "step-4k", "step-8k", "step-16k", "step-32k", "step-64k",
        "step-96k", "step-128k", "step-160k", "step-200k", "step-256k"],
       [M("validation loss", "", "val_loss", 1.2, 6.5, 3),
        M("throughput", "tokens/s", "tok_per_s", 1500, 90000),
        M("gradient norm", "", "grad_norm", 0.1, 12, 2)]),
    _t("gpus", "GPU", "GPUs",
       ["H100", "A100", "L4", "T4", "RTX 4090", "MI300X", "TPU v5e", "Gaudi 3", "V100",
        "RTX 3090", "L40S", "B200"],
       [M("utilization", "%", "util", 0, 100),
        M("temperature", "°C", "temp_c", 30, 92),
        M("memory used", "GB", "mem_gb", 0.5, 80, 1),
        M("power", "W", "power_w", 40, 1000)]),
    _t("endpoints", "API endpoint", "API endpoints",
       ["/login", "/search", "/predict", "/embed", "/upload", "/health", "/chat", "/tokenize",
        "/feedback", "/metrics", "/transcribe", "/generate"],
       [M("requests per minute", "", "rpm", 1, 6000),
        M("p95 latency", "ms", "p95_ms", 3, 2500),
        M("error rate", "%", "err_pct", 0, 12, 2)]),
    _t("products", "product", "products",
       ["headphones", "backpack", "sneakers", "water bottle", "desk lamp", "notebook",
        "phone case", "yoga mat", "keyboard", "sunglasses", "webcam", "smartwatch"],
       [M("units sold", "", "units", 0, 2500),
        M("price", "$", "price_usd", 5, 400, 2),
        M("average rating", "stars", "rating", 1, 5, 1),
        M("returns", "", "returns", 0, 120)]),
    _t("drinks", "drink", "drinks",
       ["espresso", "latte", "cappuccino", "mocha", "americano", "matcha latte", "chai",
        "cold brew", "flat white", "macchiato", "hot chocolate", "iced tea"],
       [M("cups sold", "", "cups", 0, 400),
        M("price", "$", "price_usd", 2, 7.5, 2),
        M("prep time", "s", "prep_s", 20, 300)]),
    _t("plants", "plant", "plants",
       ["tomato", "basil", "sunflower", "fern", "cactus", "orchid", "lavender", "mint", "tulip",
        "bamboo", "rosemary", "aloe vera"],
       [M("height", "cm", "height_cm", 2, 250, 1),
        M("water per week", "ml", "water_ml", 20, 3000),
        M("sunlight", "hours/day", "sun_hours", 1, 12, 1)]),
    _t("sensors", "robot sensor", "robot sensors",
       ["lidar", "camera", "IMU", "radar", "ultrasonic", "GPS", "thermometer", "microphone",
        "barometer", "wheel encoder", "depth camera", "force sensor"],
       [M("readings per second", "Hz", "rate_hz", 1, 1000),
        M("noise level", "", "noise", 0.001, 0.5, 3),
        M("battery drain", "mA", "drain_ma", 1, 900)]),
    _t("languages", "language", "languages",
       ["English", "Mandarin", "Spanish", "Hindi", "Arabic", "Swahili", "French", "Japanese",
        "Portuguese", "Korean", "Bengali", "Turkish", "Yoruba"],
       [M("training tokens", "billions", "tokens_b", 0.5, 4000, 1),
        M("BLEU score", "", "bleu", 5, 55, 1),
        M("tokens per word", "", "fertility", 1.0, 4.5, 2)]),
    _t("cities", "city", "cities",
       ["Tokyo", "Lagos", "São Paulo", "Mumbai", "Berlin", "Toronto", "Sydney", "Cairo", "Seoul",
        "Mexico City", "Paris", "Nairobi", "Jakarta"],
       [M("temperature", "°C", "temp_c", -15, 42, 1),
        M("rainfall", "mm", "rain_mm", 0, 400),
        M("air quality index", "", "aqi", 5, 300),
        M("ride-share trips", "thousands", "trips_k", 1, 950)]),
    _t("instruments", "instrument", "instruments",
       ["piano", "violin", "guitar", "drums", "flute", "cello", "trumpet", "saxophone", "harp",
        "ukulele", "clarinet", "bass"],
       [M("practice time", "min", "practice_min", 0, 180),
        M("price", "$", "price_usd", 60, 9000),
        M("recordings in dataset", "", "recordings", 10, 5000)]),
    _t("fruits", "fruit", "fruits",
       ["apple", "banana", "cherry", "mango", "kiwi", "grape", "peach", "pear", "plum", "lime",
        "papaya", "blueberry"],
       [M("price", "$/kg", "price_kg", 0.8, 14, 2),
        M("sugar", "g per 100 g", "sugar_g", 1, 20, 1),
        M("units in stock", "", "stock", 0, 900)]),
)

AXES: tuple[Axis, ...] = (
    Axis("weekday", "weekdays", ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")),
    Axis("month", "months", ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                             "Oct", "Nov", "Dec")),
    Axis("quarter", "quarters", ("Q1", "Q2", "Q3", "Q4")),
    Axis("epoch", "epochs", tuple(f"epoch {i}" for i in range(1, 11))),
    Axis("training run", "training runs", tuple(f"seed {s}" for s in (0, 1, 7, 13, 42, 123, 2024))),
    Axis("benchmark", "benchmarks", ("MMLU", "GSM8K", "HumanEval", "ARC", "HellaSwag",
                                     "TruthfulQA", "MATH", "BBH")),
    Axis("frequency band", "frequency bands", ("0-250 Hz", "250-500 Hz", "500 Hz-1 kHz",
                                               "1-2 kHz", "2-4 kHz", "4-8 kHz")),
    Axis("region", "regions", ("us-east", "us-west", "eu-central", "ap-south", "sa-east")),
    Axis("cross-validation fold", "folds", tuple(f"fold {i}" for i in range(1, 9))),
    Axis("hour", "hours", tuple(f"{h:02d}:00" for h in range(24))),
)

# class labels for classification-style problems
LABEL_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sentiment", ("negative", "neutral", "positive")),
    ("speech command", ("yes", "no", "up", "down", "left", "right", "stop", "go")),
    ("emotion", ("happy", "sad", "angry", "calm", "surprised")),
    ("email class", ("ham", "spam")),
    ("image class", ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")),
    ("intent", ("book_flight", "cancel", "weather", "greeting", "play_music", "set_alarm")),
    ("sound event", ("dog_bark", "siren", "rain", "speech", "music", "engine")),
    ("toxicity label", ("safe", "borderline", "toxic")),
    ("pet type", ("dog", "cat", "rabbit", "parrot")),
    ("defect type", ("ok", "scratch", "dent", "crack")),
)


# ---------------------------------------------------------------- helpers

def theme(rng: random.Random, min_items: int = 0) -> Theme:
    return rng.choice([t for t in THEMES if len(t.items) >= min_items])


def measure(rng: random.Random, th: Theme) -> Measure:
    return rng.choice(th.measures)


def axis(rng: random.Random, min_labels: int = 0, max_labels: int = 99) -> Axis:
    return rng.choice([a for a in AXES if min_labels <= len(a.labels) <= max_labels])


def labels(rng: random.Random, min_classes: int = 2) -> tuple[str, tuple[str, ...]]:
    return rng.choice([ls for ls in LABEL_SETS if len(ls[1]) >= min_classes])


def items(rng: random.Random, th: Theme, n: int) -> list[str]:
    return rng.sample(th.items, n)


def value(rng: random.Random, m: Measure, lo: float | None = None, hi: float | None = None):
    lo = m.lo if lo is None else lo
    hi = m.hi if hi is None else hi
    if m.decimals == 0:
        return rng.randint(int(lo), int(hi))
    return round(rng.uniform(lo, hi), m.decimals)


def values(rng: random.Random, m: Measure, n: int, distinct: bool = False) -> list:
    out: list = []
    seen: set = set()
    for _ in range(50 * n):
        if len(out) == n:
            break
        v = value(rng, m)
        if distinct and v in seen:
            continue
        seen.add(v)
        out.append(v)
    if len(out) < n:  # tiny ranges: fall back to a spread
        out = [round(m.lo + i * (m.hi - m.lo) / max(n - 1, 1), max(m.decimals, 2)) for i in range(n)]
        rng.shuffle(out)
    return out


def threshold(rng: random.Random, vals: list, m: Measure):
    """A cut-off strictly inside the data so masks are never all-True or all-False."""
    s = sorted(set(vals))
    if len(s) < 2:
        return s[0]
    i = rng.randrange(1, len(s))
    lo, hi = s[i - 1], s[i]
    t = (lo + hi) / 2
    return int(t) if m.decimals == 0 and float(t).is_integer() else round(t, m.decimals + 1)


def ident(name: str) -> str:
    """Make a safe Python identifier from a label."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_") or "x"
    return out if not out[0].isdigit() else "v_" + out
