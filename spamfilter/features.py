"""Feature engineering: text + simple metadata signals that real spam tends to carry."""
import re
import json
from email.utils import parseaddr
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_RE = re.compile(r"<[^>]+>")
MONEY_RE = re.compile(r"[$€£]\s?\d|\d+\s?(usd|birr|etb)\b", re.I)
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at",
}
RISKY_ATTACHMENT_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".com", ".pif", ".js", ".vbs",
    ".hta", ".html", ".htm", ".lnk", ".iso", ".msi",
}
OFFICE_EXTENSIONS = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptm", ".xlsm", ".docm"}


def clean_text(subject: str, body: str) -> str:
    """Subject is repeated so it carries more weight than a single body word."""
    body = HTML_RE.sub(" ", body or "")
    body = URL_RE.sub(" URLTOKEN ", body)
    subject = URL_RE.sub(" URLTOKEN ", subject or "")
    return f"{subject} {subject} {body}".strip()


def sender_address(sender):
    sender = str(sender or "")
    return parseaddr(sender)[1].strip().lower()


def sender_domain(sender):
    address = sender_address(sender)
    return address.rsplit("@", 1)[1].lower() if "@" in address else ""


def extract_display_name(sender):
    name, address = parseaddr(str(sender or ""))
    return name.strip(), address.strip().lower()


def normalize_domain(domain):
    domain = str(domain or "").strip().lower().rstrip(".")
    if domain.startswith("www.") or domain.startswith("*"):
        domain = domain[4:] if domain.startswith("www.") else domain[1:]
    return domain


def domain_is_lookalike(domain, trusted_domains):
    """Catches common character substitution and extra-character look-alike domains."""
    domain = normalize_domain(domain)
    for trusted in trusted_domains:
        trusted = normalize_domain(trusted)
        if not domain or not trusted or domain == trusted:
            continue
        # Only simple, conservative checks are used to avoid false positives.
        if domain.endswith("." + trusted) or trusted.endswith("." + domain):
            continue
        condensed_domain = re.sub(r"[^a-z]", "", domain)
        condensed_trusted = re.sub(r"[^a-z]", "", trusted)
        if condensed_domain and (
            condensed_domain.startswith(condensed_trusted)
            or condensed_trusted.startswith(condensed_domain)
        ):
            return True
        substitutions = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"}
        substituted = "".join(substitutions.get(c, c) for c in domain)
        if substituted == trusted:
            return True
    return False


def parse_authentication(headers):
    if isinstance(headers, dict):
        values = headers
    else:
        values = {}
        for line in str(headers or "").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip().lower()] = value.strip()
    return tuple(str(values.get(name, "")).strip().lower() for name in ("spf", "authentication-results-spf"))


def authentication_scores(headers):
    if isinstance(headers, dict):
        spf = headers.get("spf", headers.get("Authentication-Results-SPF", ""))
        dkim = headers.get("dkim", headers.get("DKIM-Signature", ""))
        dmarc = headers.get("dmarc", headers.get("Authentication-Results-DMARC", ""))
    else:
        text = str(headers or "")
        spf = dkim = dmarc = ""
        for line in text.splitlines():
            lower = line.lower()
            if lower.startswith("spf:"):
                spf = line.split(":", 1)[1]
            elif lower.startswith("dkim:"):
                dkim = line.split(":", 1)[1]
            elif lower.startswith("dmarc:"):
                dmarc = line.split(":", 1)[1]
            elif "spf=" in lower:
                spf = lower.split("spf=", 1)[1].split()[0]
            elif "dkim=" in lower:
                dkim = lower.split("dkim=", 1)[1].split()[0]
            elif "dmarc=" in lower:
                dmarc = lower.split("dmarc=", 1)[1].split()[0]

    def score(value, kind):
        value = str(value or "").lower()
        if kind == "dmarc":
            if "pass" in value:
                return 1.0
            if "fail" in value or "reject" in value:
                return -1.0
        else:
            if "pass" in value:
                return 1.0
            if "fail" in value or "softfail" in value or "permerror" in value:
                return -1.0
        return 0.0

    return score(spf, "spf"), score(dkim, "dkim"), score(dmarc, "dmarc")


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Numeric, sender, authentication and attachment signals."""

    def __init__(self, feature_count=None):
        self.feature_count = feature_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for i, (subject, body) in enumerate(zip(X["subject"].fillna(""), X["body"].fillna(""))):
            text = f"{subject} {body}"
            letters = [c for c in text if c.isalpha()]
            caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
            sender = str(X["sender"].iloc[i]) if "sender" in X.columns else ""
            display_name = str(X["display_name"].iloc[i]) if "display_name" in X.columns else extract_display_name(sender)[0]
            reply_to = str(X["reply_to"].iloc[i]) if "reply_to" in X.columns else ""
            channel = str(X["channel"].iloc[i] if "channel" in X.columns else "email").lower()
            domain = sender_domain(sender)
            trusted_domains = ("enatbank.com",)
            is_internal = 1.0 if domain == "enatbank.com" or domain.endswith(".enatbank.com") else 0.0
            display_mismatch = 1.0 if ("enat bank" in display_name.lower() and not is_internal) else 0.0
            reply_mismatch = 1.0 if reply_to and sender_address(reply_to) != sender_address(sender) else 0.0

            links = X["links"].iloc[i] if "links" in X.columns else URL_RE.findall(text)
            if isinstance(links, str):
                try:
                    parsed_links = json.loads(links)
                    links = parsed_links if isinstance(parsed_links, list) else re.findall(r"https?://[^\s,]+|www\.[^\s,]+", links)
                except json.JSONDecodeError:
                    links = re.findall(r"https?://[^\s,]+|www\.[^\s,]+", links)
            link_domains = []
            for link in links or []:
                parsed = urlparse(str(link) if str(link).startswith(("http://", "https://")) else "http://" + str(link))
                if parsed.hostname:
                    link_domains.append(normalize_domain(parsed.hostname))
            shortened = 1.0 if any(domain in SHORTENER_DOMAINS for domain in link_domains) else 0.0
            mismatched_links = 1.0 if any(
                ld and domain and ld != domain and not ld.endswith("." + domain) for ld in link_domains
            ) else 0.0
            lookalike = 1.0 if any(domain_is_lookalike(ld, trusted_domains) for ld in link_domains) else 0.0

            headers = X["headers"].iloc[i] if "headers" in X.columns else {
                "spf": X["spf"].iloc[i] if "spf" in X.columns else "",
                "dkim": X["dkim"].iloc[i] if "dkim" in X.columns else "",
                "dmarc": X["dmarc"].iloc[i] if "dmarc" in X.columns else "",
            }
            if isinstance(headers, str):
                try:
                    headers = json.loads(headers)
                except json.JSONDecodeError:
                    pass
            spf, dkim, dmarc = authentication_scores(headers)
            seen = float(X["sender_seen_count"].iloc[i]) if "sender_seen_count" in X.columns else 0.0
            seen = min(seen, 1000.0) / 1000.0
            redirects = float(X["redirect_count"].iloc[i]) if "redirect_count" in X.columns else 0.0
            attachments = X["attachments"].iloc[i] if "attachments" in X.columns else []
            if isinstance(attachments, str):
                attachments = [p.strip() for p in attachments.split(",") if p.strip()]
            names = [str(a) for a in attachments or []]
            extensions = [PathName(name).suffix.lower() for name in names]
            risky_attachment = 1.0 if any(ext in RISKY_ATTACHMENT_EXTENSIONS for ext in extensions) else 0.0
            macro_attachment = 1.0 if any(
                ext in {".docm", ".xlsm", ".pptm"} for ext in extensions
            ) else 0.0
            password_protected = 1.0 if any(
                "password" in str(a).lower() or "protected" in str(a).lower() for a in attachments or []
            ) else 0.0

            rows.append([
                min(len(URL_RE.findall(text)), 10) / 10,
                caps,
                min(text.count("!"), 10) / 10,
                1.0 if MONEY_RE.search(text) else 0.0,
                1.0 if HTML_RE.search(body) else 0.0,
                min(np.log1p(len(text)) / 10, 1.0),
                1.0 if not subject.strip() else 0.0,
                spf, dkim, dmarc,
                is_internal, seen, display_mismatch, reply_mismatch,
                shortened, mismatched_links, lookalike,
                min(redirects, 10.0) / 10.0,
                risky_attachment, macro_attachment, password_protected,
                1.0 if channel == "sms" else 0.0,
                1.0 if channel == "email" else 0.0,
            ])
        features = np.array(rows)
        return features[:, :self.feature_count] if self.feature_count is not None else features


class PathName(str):
    """Small filename helper kept dependency-free."""
    @property
    def suffix(self):
        name = str(self).lower()
        match = re.search(r"(\.[a-z0-9]{1,10})$", name)
        return match.group(1) if match else ""