# utils/features.py

import re
import math
from collections import Counter
from urllib.parse import urlparse

import tldextract  # pip install tldextract

# Kaggle feature order used in training
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "has_https",
    "https_in_domain",
    "contains_ip",
    "contains_at",
    "contains_double_slash",
    "url_entropy",
    "url_token_count",
    "is_shortener",
]

SHORTENERS = [
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "rebrand.ly"
]

ip_pattern = re.compile(
    r"^(https?://)?(\d{1,3}\.){3}\d{1,3}(/|:|$)",
    re.IGNORECASE
)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def extract_url_features(url: str) -> dict:
    """
    Return a dict of lexical URL features.
    EXACTLY matches the extraction you used in Colab.
    """
    url = url.strip()

    parts = tldextract.extract(url)
    domain = ""
    if parts.domain:
        domain = parts.domain
        if parts.suffix:
            domain = f"{parts.domain}.{parts.suffix}"

    subdomain = parts.subdomain or ""
    hostname = domain if not subdomain else f"{subdomain}.{domain}"

    features = {}
    rest = url.split("://", 1)[-1]

    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(rest.split("/", 1)[1]) if "/" in rest else 0
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special_chars"] = sum(1 for c in url if c in "@#?%=&_")
    features["num_subdomains"] = len(subdomain.split(".")) if subdomain else 0
    features["has_https"] = 1 if url.lower().startswith("https://") else 0
    features["https_in_domain"] = 1 if "https" in hostname.lower() else 0
    features["contains_ip"] = 1 if ip_pattern.match(url) else 0
    features["contains_at"] = 1 if "@" in url else 0
    features["contains_double_slash"] = 1 if "//" in rest else 0
    features["url_entropy"] = shannon_entropy(url)
    tokens = re.split(r"[./?=&_-]", rest)
    tokens = [t for t in tokens if t]
    features["url_token_count"] = len(tokens)
    features["is_shortener"] = 1 if any(s in hostname.lower() for s in SHORTENERS) else 0

    return features
