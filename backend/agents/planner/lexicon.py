"""Shared semantic vocabulary for test-intent parsing and element classification.

Both the requirement analyzer (understanding instructions) and the DOM
classifier (understanding pages) map onto the same *purpose* vocabulary, which
is what lets "Enter username" be matched to a field labelled "Email address".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Purpose:
    name: str
    phrases: tuple[str, ...]
    roles: tuple[str, ...]
    input_types: tuple[str, ...] = ()
    autocomplete: tuple[str, ...] = ()
    sensitive: bool = False
    related: tuple[str, ...] = ()  # purposes that are acceptable substitutes
    kind: str = "field"  # field | action | toggle | choice | page | content


PURPOSES: dict[str, Purpose] = {
    p.name: p
    for p in [
        Purpose("username_or_email", ("username", "user name", "user id", "userid", "login id", "login name",
                                      "login", "account name", "account id", "user", "email or username",
                                      "username or email", "email address or username", "identifier"),
                ("textbox",), ("text", "email"), ("username", "email"), related=("email",)),
        Purpose("email", ("email", "e-mail", "email address", "e-mail address", "mail", "email id"),
                ("textbox",), ("email",), ("email",), related=("username_or_email",)),
        Purpose("password", ("password", "passcode", "pass word", "pwd", "pass phrase", "passphrase",
                             "current password", "new password"),
                ("textbox",), ("password",), ("current-password", "new-password"), sensitive=True),
        Purpose("confirm_password", ("confirm password", "repeat password", "re-enter password", "reenter password",
                                     "retype password", "password confirmation", "verify password", "confirm"),
                ("textbox",), ("password",), ("new-password",), sensitive=True, related=("password",)),
        Purpose("full_name", ("full name", "name", "your name", "complete name", "person's name", "person name"),
                ("textbox",), ("text",), ("name",), related=("first_name",)),
        Purpose("first_name", ("first name", "given name", "forename"), ("textbox",), ("text",), ("given-name",)),
        Purpose("last_name", ("last name", "surname", "family name"), ("textbox",), ("text",), ("family-name",)),
        Purpose("phone", ("phone", "phone number", "mobile", "mobile number", "telephone", "cell", "contact number",
                          "tel"), ("textbox",), ("tel",), ("tel",)),
        Purpose("address", ("address", "street", "street address", "address line"), ("textbox",), ("text",),
                ("street-address", "address-line1")),
        Purpose("city", ("city", "town"), ("textbox", "combobox"), ("text",), ("address-level2",)),
        Purpose("state", ("state", "province", "region"), ("combobox", "textbox"), ("text",), ("address-level1",)),
        Purpose("country", ("country", "nation", "country/region"), ("combobox", "textbox"), ("text",),
                ("country", "country-name")),
        Purpose("postal_code", ("zip", "zip code", "postal code", "postcode", "pin code", "pincode"), ("textbox",),
                ("text",), ("postal-code",)),
        Purpose("search", ("search", "search box", "search bar", "search field", "query", "find", "keyword",
                           "keywords", "filter"), ("searchbox", "textbox", "combobox"), ("search", "text")),
        Purpose("date", ("date", "date of birth", "dob", "birth date", "birthday", "start date", "end date"),
                ("textbox",), ("date", "text"), ("bday",)),
        Purpose("company", ("company", "company name", "organization", "organisation", "business name", "employer"),
                ("textbox",), ("text",), ("organization",)),
        Purpose("message", ("message", "comment", "comments", "bio", "description", "notes", "feedback", "about"),
                ("textbox",), ("text",)),
        Purpose("quantity", ("quantity", "qty", "amount", "count"), ("spinbutton", "textbox"), ("number",)),
        Purpose("card_number", ("card number", "credit card", "credit card number", "card no", "debit card"),
                ("textbox",), ("text", "tel"), ("cc-number",), sensitive=True),
        Purpose("cvv", ("cvv", "cvc", "security code", "card code"), ("textbox",), ("text", "tel", "password"),
                ("cc-csc",), sensitive=True),
        Purpose("otp", ("otp", "one time password", "verification code", "2fa code", "mfa code", "auth code"),
                ("textbox",), ("text", "tel"), ("one-time-code",), sensitive=True),
        Purpose("submit_login", ("login", "log in", "sign in", "signin", "log on", "logon", "sign on",
                                 "login button", "sign in button"),
                ("button", "link"), kind="action", related=("submit_generic",)),
        Purpose("submit_register", ("register", "sign up", "signup", "create account", "create an account",
                                    "join", "get started", "create my account"),
                ("button", "link"), kind="action", related=("submit_generic",)),
        Purpose("submit_generic", ("submit", "save", "continue", "next", "proceed", "confirm", "apply", "send",
                                   "done", "ok", "finish", "update", "save changes"),
                ("button",), kind="action"),
        Purpose("logout", ("logout", "log out", "sign out", "signout", "log off"), ("button", "link", "menuitem"),
                kind="action"),
        Purpose("close", ("close", "dismiss", "x", "cancel", "no thanks", "not now"), ("button",), kind="action"),
        Purpose("accept_cookies", ("accept cookies", "accept all", "allow all", "agree", "got it", "i agree",
                                   "accept all cookies", "allow cookies"), ("button",), kind="action"),
        Purpose("terms", ("terms", "terms and conditions", "terms of service", "privacy policy", "i agree",
                          "accept terms", "agreement", "conditions"), ("checkbox",), kind="toggle"),
        Purpose("remember_me", ("remember me", "keep me signed in", "stay signed in", "keep me logged in",
                                "remember"), ("checkbox",), kind="toggle"),
        Purpose("forgot_password", ("forgot password", "forgot your password", "reset password",
                                    "forgotten password"), ("link", "button"), kind="action"),
        Purpose("add_to_cart", ("add to cart", "add to basket", "add to bag", "buy now"), ("button",), kind="action"),
        Purpose("checkout", ("checkout", "check out", "place order", "pay now", "proceed to checkout",
                             "complete purchase", "buy"), ("button", "link"), kind="action"),
        Purpose("delete", ("delete", "remove", "trash", "discard", "destroy"), ("button", "menuitem"), kind="action"),
        Purpose("upload", ("upload", "attach", "choose file", "browse", "file", "document", "attachment", "resume",
                           "cv", "avatar", "photo"), ("button",), ("file",), kind="field"),
        Purpose("menu", ("menu", "hamburger", "navigation menu", "main menu", "more options", "options"),
                ("button",), kind="action"),
        Purpose("settings", ("settings", "preferences", "gear", "cog", "configuration"), ("button", "link"),
                kind="action"),
    ]
}

SENSITIVE_PURPOSES = {name for name, p in PURPOSES.items() if p.sensitive}

# element kind nouns -> preferred aria roles
KIND_ROLES: dict[str, tuple[str, ...]] = {
    "button": ("button",),
    "btn": ("button",),
    "link": ("link",),
    "hyperlink": ("link",),
    "field": ("textbox", "searchbox", "spinbutton", "combobox"),
    "input": ("textbox", "searchbox", "spinbutton", "combobox"),
    "input field": ("textbox", "searchbox", "spinbutton", "combobox"),
    "text field": ("textbox", "searchbox"),
    "textfield": ("textbox", "searchbox"),
    "textbox": ("textbox", "searchbox"),
    "text box": ("textbox", "searchbox"),
    "text area": ("textbox",),
    "textarea": ("textbox",),
    "box": ("textbox", "searchbox", "checkbox"),
    "search box": ("searchbox", "textbox"),
    "search bar": ("searchbox", "textbox"),
    "dropdown": ("combobox", "listbox"),
    "drop-down": ("combobox", "listbox"),
    "drop down": ("combobox", "listbox"),
    "select": ("combobox", "listbox"),
    "select box": ("combobox", "listbox"),
    "combo box": ("combobox",),
    "combobox": ("combobox",),
    "picker": ("combobox", "button", "textbox"),
    "list": ("listbox", "list", "combobox"),
    "listbox": ("listbox",),
    "list box": ("listbox",),
    "menu": ("button", "link", "menuitem", "menu", "menubar"),
    "menu item": ("menuitem",),
    "menuitem": ("menuitem",),
    "option": ("option", "radio", "menuitem"),
    "checkbox": ("checkbox",),
    "check box": ("checkbox",),
    "tickbox": ("checkbox",),
    "tick box": ("checkbox",),
    "radio": ("radio",),
    "radio button": ("radio",),
    "toggle": ("switch", "checkbox", "button"),
    "switch": ("switch", "checkbox"),
    "tab": ("tab",),
    "icon": ("button", "link", "img"),
    "image": ("img",),
    "logo": ("img", "link"),
    "heading": ("heading",),
    "header": ("heading", "banner"),
    "title": ("heading",),
    "slider": ("slider",),
    "row": ("row",),
    "cell": ("cell", "gridcell"),
    "table": ("table", "grid"),
    "card": ("article", "generic", "listitem", "region"),
    "tile": ("article", "generic", "listitem"),
    "item": ("listitem", "option", "menuitem", "article", "generic"),
    "dialog": ("dialog", "alertdialog"),
    "modal": ("dialog", "alertdialog"),
    "popup": ("dialog", "alertdialog"),
    "pop-up": ("dialog", "alertdialog"),
    "banner": ("dialog", "region", "banner"),
    "canvas": ("canvas", "img", "generic"),
    "message": ("alert", "status", "generic", "paragraph"),
    "error": ("alert", "status", "generic", "paragraph"),
    "text": ("generic", "paragraph", "heading", "text"),
    "label": ("generic", "text"),
    "section": ("region", "generic"),
    "column": ("region", "list", "generic", "columnheader"),
    "area": ("region", "generic"),
    "zone": ("region", "generic"),
}

KIND_NOUN_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in KIND_ROLES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

ACTION_ROLE_COMPAT: dict[str, tuple[str, ...]] = {
    "click": ("button", "link", "menuitem", "tab", "checkbox", "radio", "switch", "option", "combobox",
              "menuitemcheckbox", "menuitemradio", "treeitem", "img", "generic", "cell", "gridcell", "row",
              "listitem", "heading", "article", "summary", "button-like"),
    "fill": ("textbox", "searchbox", "combobox", "spinbutton"),
    "type": ("textbox", "searchbox", "combobox", "spinbutton"),
    "clear": ("textbox", "searchbox", "combobox", "spinbutton"),
    "select": ("combobox", "listbox", "radio", "option", "menuitemradio", "tab", "button"),
    "check": ("checkbox", "switch", "radio", "menuitemcheckbox"),
    "uncheck": ("checkbox", "switch", "menuitemcheckbox"),
    "toggle": ("checkbox", "switch", "button"),
    "hover": ("button", "link", "menuitem", "img", "generic", "listitem", "tab", "heading", "cell", "article"),
    "upload": ("button", "textbox", "generic"),
    "drag": ("listitem", "generic", "article", "img", "button", "row", "option", "region"),
    "drop": ("region", "list", "generic", "listitem", "article", "group"),
    "download": ("link", "button"),
}

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "combobox", "listbox", "option", "checkbox", "radio", "switch",
    "slider", "spinbutton", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "treeitem", "gridcell",
}

CONTAINER_ROLES = {"form", "dialog", "alertdialog", "region", "navigation", "banner", "main", "contentinfo",
                   "complementary", "search", "group", "list", "table", "grid", "tablist", "menu", "menubar",
                   "toolbar", "radiogroup", "tree", "article", "listitem", "row"}

LANDMARK_ROLES = {"banner", "navigation", "main", "contentinfo", "complementary", "search", "region", "form"}

COLOR_WORDS = {"red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey",
               "brown", "cyan", "magenta", "teal", "violet"}
SHAPE_WORDS = {"circle", "square", "rectangle", "triangle", "star", "dot", "shape", "node", "marker", "point",
               "diamond", "hexagon", "ellipse", "oval"}

ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "last": -1, "final": -1}

DESTRUCTIVE_WORDS = ("delete", "remove", "destroy", "erase", "purge", "terminate", "deactivate", "close account",
                     "cancel subscription", "drop", "wipe", "revoke", "trash", "discard")
FINANCIAL_WORDS = ("pay", "purchase", "buy", "place order", "checkout", "check out", "transfer", "payment",
                   "subscribe", "donate", "withdraw", "deposit", "confirm order", "complete purchase", "billing")
MESSAGING_WORDS = ("send", "email", "message", "invite", "post", "publish", "share", "tweet", "notify",
                   "broadcast", "reply")
SUBMIT_WORDS = ("submit", "save", "register", "sign up", "create", "apply", "confirm", "update", "sign in",
                "log in", "login", "continue", "next", "proceed", "finish", "done")

STOPWORDS = {"the", "a", "an", "on", "in", "into", "to", "of", "for", "with", "your", "my", "our", "this",
             "that", "and", "or", "at", "by", "from", "is", "be", "as", "please", "then", "it"}

# A small gazetteer so "select India, Kerala and Kochi" can be mapped onto Country/State/City
# dropdowns when option lists are not yet rendered (custom dropdowns load options lazily).
GAZETTEER: dict[str, str] = {}
for _country in ("india", "united states", "usa", "united kingdom", "uk", "canada", "australia", "germany",
                 "france", "japan", "china", "brazil", "mexico", "spain", "italy", "singapore", "netherlands",
                 "south africa", "new zealand", "ireland", "sweden", "norway", "uae", "united arab emirates"):
    GAZETTEER[_country] = "country"
for _state in ("kerala", "karnataka", "tamil nadu", "maharashtra", "delhi", "goa", "gujarat", "punjab",
               "california", "texas", "new york", "florida", "washington", "ontario", "quebec", "bavaria",
               "new south wales", "victoria", "queensland", "telangana", "west bengal", "rajasthan"):
    GAZETTEER[_state] = "state"
for _city in ("kochi", "cochin", "trivandrum", "thiruvananthapuram", "bangalore", "bengaluru", "chennai",
              "mumbai", "pune", "hyderabad", "kolkata", "new delhi", "los angeles", "san francisco", "austin",
              "seattle", "new york city", "nyc", "toronto", "montreal", "sydney", "melbourne", "london",
              "paris", "berlin", "munich", "tokyo", "calicut", "kozhikode", "thrissur", "mysore"):
    GAZETTEER[_city] = "city"


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel Case
    text = re.sub(r"[^\w\s@.+/]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(text: str | None) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t and t not in STOPWORDS]


@dataclass
class PurposeMatch:
    purpose: str
    score: float
    matched_phrase: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def classify_phrase(text: str | None, *, kinds: tuple[str, ...] | None = None) -> PurposeMatch | None:
    """Map free text (an instruction target or an element label) to a purpose."""

    norm = normalize_text(text)
    if not norm:
        return None
    scored: list[tuple[str, float, str]] = []
    norm_tokens = set(norm.split())
    for purpose in PURPOSES.values():
        if kinds and purpose.kind not in kinds:
            continue
        best = 0.0
        best_phrase = ""
        for phrase in purpose.phrases:
            if norm == phrase:
                score = 1.0
            elif re.search(rf"(^|\s){re.escape(phrase)}(\s|$)", norm):
                # phrase contained in text: longer phrases are more specific
                score = 0.72 + min(0.24, 0.04 * len(phrase.split()) + 0.01 * len(phrase))
            else:
                phrase_tokens = set(phrase.split())
                overlap = len(phrase_tokens & norm_tokens)
                score = 0.5 * overlap / len(phrase_tokens) if overlap and len(phrase_tokens) > 1 else 0.0
            if score > best:
                best, best_phrase = score, phrase
        if best > 0:
            scored.append((purpose.name, best, best_phrase))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[1], -len(item[2])))
    top = scored[0]
    return PurposeMatch(purpose=top[0], score=round(top[1], 3), matched_phrase=top[2],
                        alternatives=[(n, s) for n, s, _ in scored[1:4]])


def purpose_compatible(a: str | None, b: str | None) -> float:
    """1.0 for same purpose, 0.8 for declared substitutes, 0 otherwise."""

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    pa, pb = PURPOSES.get(a), PURPOSES.get(b)
    if (pa and b in pa.related) or (pb and a in pb.related):
        return 0.8
    return 0.0


def synonyms_for(purpose: str | None) -> list[str]:
    if not purpose or purpose not in PURPOSES:
        return []
    return list(PURPOSES[purpose].phrases)


def risk_for_text(text: str) -> str | None:
    norm = normalize_text(text)
    if any(re.search(rf"\b{re.escape(w)}\b", norm) for w in FINANCIAL_WORDS):
        return "financial"
    if any(re.search(rf"\b{re.escape(w)}\b", norm) for w in DESTRUCTIVE_WORDS):
        return "destructive"
    if any(re.search(rf"\b{re.escape(w)}\b", norm) for w in MESSAGING_WORDS):
        return "messaging"
    if any(re.search(rf"\b{re.escape(w)}\b", norm) for w in SUBMIT_WORDS):
        return "submit"
    return None
