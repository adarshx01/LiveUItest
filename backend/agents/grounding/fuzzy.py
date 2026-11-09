"""Forgiving text matching for *finding* things: element labels, dropdown options, remembered strategies.

Test authors misspell ("Clik on Sing In", "Selct Indai"), use other words than the page ("Log in" for
"Sign in", "pwd" for "Password") and glue or split words ("e-mail"/"email", "user name"/"username").
``match_score`` repairs those differences against the *candidate's own words* and then reuses the
calibrated ``similarity`` scale, discounted by how much repair was needed:

* typos: a weighted Damerau-Levenshtein distance where cheap edits are the ones people make
  (transposed letters, a key next to the intended one, a dropped or doubled letter) and an arbitrary
  substitution is expensive. The allowance grows with word length, so "Sing" -> "Sign" and
  "dashbaord" -> "dashboard" repair, while "Save" never becomes "Same" (m and v are not neighbours);
* synonyms and abbreviations: curated equivalence groups ("sign in" = "log in" = "login",
  "pwd" = "password", "qty" = "quantity"). Different groups never match ("Sign In" vs "Sign Up");
* compounds: "drop down" = "dropdown", "check out" = "checkout";
* numbers and codes are exact: "Item 1" never matches "Item 7".

``text_match.similarity`` stays strict on purpose. Assertions compare expected and actual values with
it, and a forgiving comparison there would turn real defects into passes.
"""

from __future__ import annotations

import re
from functools import lru_cache

from backend.agents.grounding.text_match import similarity
from backend.agents.planner.lexicon import STOPWORDS, normalize_text

# ---------------------------------------------------------------------------- vocabulary
# Each group is one meaning. Order inside a group does not matter; groups never overlap.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("login", "log in", "sign in", "signin", "log on", "logon", "sign on"),
    ("logout", "log out", "sign out", "signout", "log off", "logoff"),
    ("register", "sign up", "signup", "create account", "create an account", "join now"),
    ("cart", "basket", "shopping cart", "shopping basket", "shopping bag"),
    ("delete", "remove", "trash"),
    ("edit", "modify"),
    ("search", "look up", "lookup"),
    ("phone", "telephone", "mobile", "phone number", "mobile number", "telephone number", "cell phone", "tel"),
    ("email", "e mail", "email address", "e mail address", "email id"),
    ("username", "user name", "user id", "userid", "login id", "login name"),
    ("password", "passcode", "pwd", "pass word"),
    ("first name", "given name", "forename", "fname"),
    ("last name", "surname", "family name", "lname"),
    ("zip", "zip code", "postal code", "postcode", "pin code", "pincode", "zipcode"),
    ("date of birth", "dob", "birth date", "birthdate", "birthday"),
    ("company", "organization", "organisation", "employer", "company name"),
    ("next", "continue", "proceed"),
    ("close", "dismiss"),
    ("settings", "preferences", "configuration", "prefs"),
    ("ok", "okay"),
    # icons are described by their shape ("three dots", "kebab") or by their meaning ("more options")
    ("three dots", "3 dots", "dots", "kebab", "kebab menu", "ellipsis", "more options", "more actions", "more",
     "options menu", "actions menu", "overflow menu", "⋮", "⋯", "…", "...", "dots vertical", "more vert",
     "more horiz", "dots horizontal"),
    ("hamburger", "hamburger menu", "three lines", "menu icon", "navigation toggle", "☰"),
    ("pencil", "pencil icon", "edit icon"),
    ("bin", "trash can", "dustbin", "garbage", "delete icon", "trash icon"),
    ("cross", "x icon", "close icon", "×", "✕"),
    ("plus", "plus icon", "add icon", "+"),
    ("magnifier", "magnifying glass", "search icon", "loupe"),
    ("gear", "cog", "gear icon", "settings icon", "cog icon"),
    ("eye", "eye icon", "view icon", "preview icon", "show"),
    ("checkout", "check out"),
    ("quantity", "qty"),
    ("message", "msg"),
    ("description", "desc"),
    ("information", "info"),
    ("address", "addr"),
    ("account", "acct"),
    ("number", "num"),
    ("dropdown", "drop down", "drop-down"),
    ("checkbox", "check box", "tickbox", "tick box"),
)

# country / region aliases used for dropdown options ("Select USA" -> "United States")
OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "united states": ("usa", "us", "u s", "u s a", "america", "united states of america", "the united states"),
    "united kingdom": ("uk", "u k", "britain", "great britain", "gb", "the united kingdom"),
    "united arab emirates": ("uae", "u a e", "emirates"),
    "netherlands": ("holland", "the netherlands"),
    "south korea": ("korea", "republic of korea"),
    "russia": ("russian federation",),
    "czech republic": ("czechia",),
}

GENERIC_WORDS = {"button", "btn", "link", "field", "input", "box", "icon", "click", "here", "tab", "option",
                 "dropdown", "menu", "item", "text", "control", "entry", "area", "value"}

# Correctly spelled words that commonly appear in UI labels and test steps. Two *known* words are never
# treated as typos of each other ("Edit" is not "Exit", "Save" is not "Sale", "from" is not "form"); a
# misspelling on either side ("Sumbit", "Indai", "Sing In") still repairs.
_UI_WORDS = """
about accept access account accounts action actions activate active add added address admin advanced agree alert
alerts all allow already amount and answer apply approve archive area ascending assign attach audio auto available
back bank basket before bell below billing bin birth block blog book books bottom browse buy calendar call cancel
card cards career careers cart category change chat check checkout choose city clear click close code collapse color
colour comment comments company compare confirm connect contact continue copy country coupon create created credit
current customer customers cut dark dashboard data date day days decline default delete deliver delivery deny
descending description detail details disable disabled discount dismiss display done down download draft drafts drag
drop edit email employee enable enabled end english enter error event events exit expand export favorite favourite
feature features feedback field file files filter find first folder follow form forgot forward free from full gallery
general get gift give group groups guest help hide high history home hour image images import inbox info information
invite invoice item items join just keep key label language last later learn leave left less light like limit line
link links list live load loading local location lock log login logout low mail main manage map mark medium member
members menu message messages method middle mode month more move music name navigate new news next none not note
notes notification notifications now number off offer offers old once only open option options order orders other
out over overview owner page pages paid password pause pay payment pending people phone photo photos pin plan plans
play please policy post posts preferences prev preview previous price pricing print privacy product products profile
project projects public publish quantity question quick rate read ready recent record redo refresh region register
reject remember remind remove rename reply report reports request required reset restore retry return review right
role roles row rule run sale same save saved schedule search second secure security see select send sent service
services session set settings setup share shipping shop show sign signup site size skip small sort source spam
start state status step stop store street submit subscribe success summary support surname switch sync table tag
tags task tasks tax team terms test text theme time title today top total track transfer trash try type undo unit
unlock update upgrade upload user username users value version video view visit wallet week welcome window wish
wishlist with word work year yes zip
"""


def _build_known_words() -> frozenset[str]:
    from backend.agents.planner.lexicon import KIND_ROLES, PURPOSES

    words = set(_UI_WORDS.split())
    for purpose in PURPOSES.values():
        for phrase in purpose.phrases:
            words.update(w for w in normalize_text(phrase).split() if len(w) >= 3)
    for kind in KIND_ROLES:
        words.update(normalize_text(kind).split())
    for group in SYNONYM_GROUPS:
        for phrase in group:
            words.update(w for w in normalize_text(phrase).split() if len(w) >= 3)
    return frozenset(words)


KNOWN_WORDS = _build_known_words()

# QWERTY neighbours: a substitution between adjacent keys is a likely slip of the finger
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_POS = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row)}


def _adjacent(a: str, b: str) -> bool:
    pa, pb = _POS.get(a), _POS.get(b)
    if pa is None or pb is None:
        return False
    return abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1 and pa != pb


COST_TRANSPOSE = 0.5
COST_ADJACENT = 0.6
COST_INDEL = 0.8
COST_SUBSTITUTE = 1.0


def typo_cost(a: str, b: str, limit: float = 3.0) -> float:
    """Weighted optimal-string-alignment distance (Damerau-Levenshtein without repeated edits)."""

    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if abs(la - lb) * COST_INDEL > limit:
        return limit + 1
    prev2: list[float] = []
    prev = [j * COST_INDEL for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [i * COST_INDEL] + [0.0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                sub = prev[j - 1]
            else:
                sub = prev[j - 1] + (COST_ADJACENT if _adjacent(a[i - 1], b[j - 1]) else COST_SUBSTITUTE)
            cost = min(prev[j] + COST_INDEL, cur[j - 1] + COST_INDEL, sub)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and prev2:
                cost = min(cost, prev2[j - 2] + COST_TRANSPOSE)
            cur[j] = cost
            row_min = min(row_min, cost)
        if row_min > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb]


def typo_allowance(length: int) -> float:
    """Budget of edit cost a word of this length may need and still count as the same word."""

    if length <= 3:
        return COST_TRANSPOSE  # "teh" -> "the", but "sav" stays "sav"
    if length <= 5:
        return COST_INDEL  # one cheap slip: "Loign", "Logn", "Sace"; never an arbitrary substitution
    if length <= 8:
        return 2 * COST_INDEL  # "pasword", "usrname", "dashbaord", "contry"
    return 3 * COST_INDEL


@lru_cache(maxsize=200_000)
def word_typo_cost(query: str, candidate: str) -> float | None:
    """Repair cost from ``query`` to ``candidate`` if they plausibly are the same word, else None."""

    if query == candidate:
        return 0.0
    if any(ch.isdigit() for ch in query + candidate) or query.startswith("zzq") or candidate.startswith("zzq"):
        return None  # numbers, codes and canonical synonym tokens are exact
    if query in KNOWN_WORDS and candidate in KNOWN_WORDS:
        return None  # two real words ("edit"/"exit"): a different word, not a typo
    if query[0] != candidate[0] and not (len(query) > 1 and len(candidate) > 1 and query[0] == candidate[1]
                                         and query[1] == candidate[0]):
        return None  # people rarely mistype the first letter, except by swapping the first two
    budget = typo_allowance(max(len(query), len(candidate)))
    cost = typo_cost(query, candidate, budget)
    if query[0] != candidate[0]:
        cost = max(cost, COST_INDEL)  # swapping the first two letters is a rarer slip than a dropped letter
    return cost if cost <= budget else None


# ---------------------------------------------------------------------------- canonical forms
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\S){re.escape(phrase)}(?!\S)")


CANONICAL_PREFIX = "zzq"
_TOKEN_ALPHABET = "bcfhjkmnprtvwx"  # no letters that start a stemmable suffix (s, e, d, g)


def _group_token(index: int) -> str:
    """Opaque, mutually dissimilar token per synonym group (no digits: similarity() compares them as words)."""

    letters = []
    value = index * 7919 + 104729  # spread consecutive groups apart
    for _ in range(5):
        value, rem = divmod(value, len(_TOKEN_ALPHABET))
        letters.append(_TOKEN_ALPHABET[rem])
    return CANONICAL_PREFIX + "".join(letters) + "q"


_CANONICAL_RULES: list[tuple[re.Pattern[str], str]] = []
for _index, _group in enumerate(SYNONYM_GROUPS):
    for _phrase in _group:
        _norm = normalize_text(_phrase)
        if _norm:
            _CANONICAL_RULES.append((_phrase_pattern(_norm), _group_token(_index)))
_CANONICAL_RULES.sort(key=lambda rule: -len(rule[0].pattern))  # longest phrases first

# words that carry synonym meaning; a typo of one of them is repaired before canonicalization
_SYNONYM_VOCAB: tuple[str, ...] = tuple(sorted({w for g in SYNONYM_GROUPS for p in g
                                                for w in normalize_text(p).split() if len(w) >= 4}))
_SYNONYM_VOCAB_SET = frozenset(_SYNONYM_VOCAB)


@lru_cache(maxsize=50_000)
def canonical(text: str) -> str:
    """Normalized text with every synonym/abbreviation phrase replaced by its group token."""

    out = normalize_text(text)
    if not out:
        return out
    for pattern, token in _CANONICAL_RULES:
        if pattern.search(out):
            out = pattern.sub(token, out)
    return out


def _words(text: str) -> list[str]:
    return [w for w in text.split() if w]


@lru_cache(maxsize=50_000)
def repair(query: str, candidate: str) -> tuple[str, float]:
    """Rewrite ``query`` (canonical form) with the candidate's spelling wherever a word is a typo of one of
    the candidate's words, or a split/glued form of them. Returns the repaired text and the edit cost."""

    q_words, c_words = _words(query), _words(candidate)
    if not q_words or not c_words:
        return query, 0.0
    c_set = set(c_words)
    joined_pairs = {c_words[i] + c_words[i + 1]: f"{c_words[i]} {c_words[i + 1]}" for i in range(len(c_words) - 1)}
    out: list[str] = []
    cost = 0.0
    i = 0
    while i < len(q_words):
        word = q_words[i]
        if word in c_set or word in STOPWORDS:
            out.append(word)
            i += 1
            continue
        # glued in the query, split on the page: "dropdown" vs "drop down"
        if word in joined_pairs:
            out.append(joined_pairs[word])
            cost += 0.2
            i += 1
            continue
        # split in the query, glued on the page: "user name" vs "username"
        if i + 1 < len(q_words) and word + q_words[i + 1] in c_set:
            out.append(word + q_words[i + 1])
            cost += 0.2
            i += 2
            continue
        options = sorted((c, cand) for cand in set(c_words) if (c := word_typo_cost(word, cand)) is not None)
        if options and not (len(options) > 1 and options[1][0] <= options[0][0] + AMBIGUITY_WINDOW / 2):
            out.append(options[0][1])
            cost += options[0][0]
        else:
            out.append(word)
        i += 1
    return " ".join(out), round(cost, 3)


AMBIGUITY_WINDOW = 0.35  # repairs this close in cost are equally plausible readings


def _discount(cost: float) -> float:
    return max(0.75, 1.0 - 0.06 * cost)


@lru_cache(maxsize=50_000)
def vocab_repair(text: str) -> tuple[str, float]:
    """Fix typos in words that carry synonym meaning ("sing in" -> "sign in", "pasword" -> "password")."""

    words = _words(text)
    out: list[str] = []
    cost = 0.0
    for word in words:
        if len(word) < 4 or word in STOPWORDS or word in _SYNONYM_VOCAB_SET or word.startswith("zzq"):
            out.append(word)
            continue
        best: list[tuple[float, str]] = []
        for vocab in _SYNONYM_VOCAB:
            c = word_typo_cost(word, vocab)
            if c is not None:
                best.append((c, vocab))
        if not best:
            out.append(word)
            continue
        best.sort()
        close = {v for c, v in best if c <= best[0][0] + AMBIGUITY_WINDOW}
        if len({canonical(v) for v in close}) > 1:
            out.append(word)  # "usrname": username or surname? different meanings - do not guess
            continue
        out.append(best[0][1])
        cost += best[0][0]
    return " ".join(out), round(cost, 3)


@lru_cache(maxsize=200_000)
def match_score(query: str, candidate: str) -> float:
    """How well ``candidate`` (page text) matches ``query`` (what the test asked for), in [0, 1].

    Identical to ``similarity`` for clean input; repairs typos, synonyms, abbreviations and
    compounds otherwise, with a discount per repair so an exact match always outranks a repaired one.
    """

    base = similarity(query, candidate)
    if base >= 0.97 or not query or not candidate:
        return base
    nq, nc = normalize_text(query), normalize_text(candidate)
    best = base
    # 1. typos against the candidate's own words ("Loign" -> "Login", "Indai" -> "India")
    repaired, cost = repair(nq, nc)
    if cost > 0 and repaired != nq:
        best = max(best, similarity(repaired, nc) * _discount(cost))
    # 2. synonyms and abbreviations, after fixing typos inside synonym words ("Sing in" -> "sign in" = "Log in")
    fixed, vocab_cost = vocab_repair(repaired)
    cq, cc = canonical(fixed), canonical(nc)
    if (cq, cc) != (fixed, nc) and best < 0.6:
        # synonyms only bridge *different* wording ("Sign in" / "Log in"); when the literal words already
        # overlap ("Cart" / "Add to cart") they would merely distort the length-based containment score
        repaired_canonical, cost2 = repair(cq, cc)
        total = cost + vocab_cost + cost2
        best = max(best, similarity(repaired_canonical, cc) * 0.96 * _discount(total))
    return round(min(best, 0.99), 4)


def explain_match(query: str, candidate: str) -> str | None:
    """Human-readable reason when ``match_score`` needed repairs (for evidence and reports)."""

    nq, nc = normalize_text(query), normalize_text(candidate)
    if nq == nc or similarity(query, candidate) >= 0.97:
        return None
    notes: list[str] = []
    repaired, _ = repair(nq, nc)
    fixed, _ = vocab_repair(repaired)
    fixes = [f"{a}->{b}" for a, b in zip(nq.split(), fixed.split(), strict=False) if a != b]
    if fixes and len(nq.split()) == len(fixed.split()):
        notes.append("typo " + ", ".join(fixes[:3]))
    if canonical(fixed) != fixed or canonical(nc) != nc:
        if similarity(canonical(fixed), canonical(nc)) > similarity(fixed, nc):
            notes.append("synonym/abbreviation")
    return "; ".join(notes) or None


def best_match(query: str, candidates: list[str], *, min_score: float = 0.8,
               min_margin: float = 0.06) -> tuple[int | None, float, float]:
    """Index of the unique best candidate (score >= min_score and clear of the runner-up by min_margin)."""

    scored = sorted(((match_score(query, c), i) for i, c in enumerate(candidates) if c), reverse=True)
    if not scored or scored[0][0] < min_score:
        return None, scored[0][0] if scored else 0.0, 0.0
    margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
    if margin < min_margin and normalize_text(candidates[scored[0][1]]) != normalize_text(candidates[scored[1][1]]):
        return None, scored[0][0], margin
    return scored[0][1], scored[0][0], margin


def option_aliases(text: str) -> list[str]:
    norm = normalize_text(text)
    out: list[str] = []
    for full, aliases in OPTION_ALIASES.items():
        if norm == full or norm in aliases:
            out.extend([full, *aliases])
    return [a for a in dict.fromkeys(out) if a != norm]


def token_weight(word: str) -> float:
    if any(ch.isdigit() for ch in word):
        return 1.5
    if word in GENERIC_WORDS or word in STOPWORDS:
        return 0.3
    return 1.0
