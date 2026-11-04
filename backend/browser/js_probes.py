"""Registry of platform-authored, read-only page probes.

``browser_evaluate`` can run arbitrary JavaScript, so the adapter only accepts
functions built here. Probes read state; the few that move the viewport are
marked ``mutates_view``. Parameters are embedded with ``json.dumps`` so page or
test data can never break out of the literal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    source: str
    element: bool = False  # function receives the target element
    mutates_view: bool = False


_CSS_PATH_JS = r"""
const cssEscape = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
const looksGenerated = (s) => !s || /\d{4,}|[a-f0-9]{8,}|^(ember|react|mui|css|sc|jsx|ng|_)[-_]?/i.test(s);
const uniq = (sel) => { try { return document.querySelectorAll(sel).length === 1; } catch (e) { return false; } };
const cssPath = (el) => {
  if (!(el instanceof Element)) return null;
  const tid = el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa');
  if (tid) { const s = `[data-testid="${tid}"]`; if (el.getAttribute('data-testid') && uniq(s)) return s; }
  if (el.id && !looksGenerated(el.id)) { const s = '#' + cssEscape(el.id); if (uniq(s)) return s; }
  const nm = el.getAttribute('name');
  if (nm) { const s = `${el.tagName.toLowerCase()}[name="${nm}"]`; if (uniq(s)) return s; }
  const parts = [];
  let cur = el;
  for (let depth = 0; cur && cur.nodeType === 1 && depth < 6; depth++) {
    let part = cur.tagName.toLowerCase();
    if (cur.id && !looksGenerated(cur.id)) { parts.unshift('#' + cssEscape(cur.id)); break; }
    const parent = cur.parentElement;
    if (parent) {
      const same = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
      if (same.length > 1) part += `:nth-of-type(${same.indexOf(cur) + 1})`;
    }
    parts.unshift(part);
    cur = parent;
  }
  return parts.join(' > ');
};
"""

_LABEL_JS = r"""
const textOf = (n) => (n ? (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim() : '');
const labelFor = (el) => {
  const by = el.getAttribute('aria-labelledby');
  if (by) { const t = by.split(/\s+/).map(id => textOf(document.getElementById(id))).join(' ').trim(); if (t) return t; }
  if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
  if (el.labels && el.labels.length) return textOf(el.labels[0]);
  const wrap = el.closest('label'); if (wrap) return textOf(wrap);
  return '';
};
"""

DOM_INVENTORY = Probe("dom_inventory", r"""() => {
""" + _CSS_PATH_JS + _LABEL_JS + r"""
  const MAX = 700;
  const vw = window.innerWidth, vh = window.innerHeight;
  const sel = 'a[href],button,input,select,textarea,summary,details,[role],[tabindex],[onclick],[contenteditable="true"],' +
              '[contenteditable=""],canvas,svg[aria-label],img,[data-testid],[draggable="true"],iframe,label,h1,h2,h3,h4,form,dialog,[aria-modal="true"]';
  const seen = new Set();
  const nodes = [];
  for (const el of document.querySelectorAll(sel)) { if (!seen.has(el)) { seen.add(el); nodes.push(el); } }
  // clickable-looking elements without semantics (cursor:pointer) - bounded scan
  let scanned = 0;
  for (const el of document.body ? document.body.querySelectorAll('div,span,li,i,svg,td,p') : []) {
    if (scanned++ > 4000) break;
    if (seen.has(el)) continue;
    const st = getComputedStyle(el);
    if (st.cursor === 'pointer' && !(el.parentElement && getComputedStyle(el.parentElement).cursor === 'pointer')) {
      seen.add(el); nodes.push(el);
    }
  }
  const out = [];
  for (const el of nodes) {
    if (out.length >= MAX) break;
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    const visible = r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none' && parseFloat(st.opacity || '1') > 0.02;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let obstructed = false;
    if (visible) {
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      if (cx >= 0 && cy >= 0 && cx < vw && cy < vh) {
        const top = document.elementFromPoint(cx, cy);
        obstructed = !!top && top !== el && !el.contains(top) && !top.contains(el) && !(el.labels && Array.from(el.labels).some(l => l.contains(top)));
      }
    }
    const form = el.form || el.closest('form');
    const fieldset = el.closest('fieldset');
    let heading = '';
    for (let cur = el; cur && !heading; cur = cur.parentElement) {
      const h = cur.querySelector ? cur.querySelector(':scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > legend') : null;
      if (h && h !== el) heading = textOf(h).slice(0, 80);
    }
    const iconHint = [el.getAttribute('class') || '', el.querySelector && el.querySelector('svg title') ? textOf(el.querySelector('svg title')) : '',
                      el.querySelector && el.querySelector('i[class]') ? el.querySelector('i[class]').className : ''].join(' ')
                      .split(/\s+/).filter(c => /icon|fa-|bi-|material|glyph|gear|cog|trash|close|menu|search|settings|edit|delete|plus|cart/i.test(c)).slice(0, 4).join(' ');
    const item = {
      tag, type, id: el.id || null, name: el.getAttribute('name'), testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa'),
      role: el.getAttribute('role'), ariaLabel: el.getAttribute('aria-label'), title: el.getAttribute('title'),
      placeholder: el.getAttribute('placeholder'), autocomplete: el.getAttribute('autocomplete'), label: labelFor(el).slice(0, 120),
      text: textOf(el).slice(0, 120), href: el.getAttribute('href'), alt: el.getAttribute('alt'),
      value: (tag === 'input' || tag === 'textarea' || tag === 'select') && type !== 'password' && type !== 'file' ? String(el.value || '').slice(0, 120) : null,
      valueLength: (tag === 'input' || tag === 'textarea') ? String(el.value || '').length : null,
      checked: (type === 'checkbox' || type === 'radio') ? !!el.checked : (el.getAttribute('aria-checked') != null ? el.getAttribute('aria-checked') === 'true' : null),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true', readonly: !!el.readOnly, required: !!el.required || el.getAttribute('aria-required') === 'true',
      expanded: el.getAttribute('aria-expanded'), modal: el.getAttribute('aria-modal') === 'true' || (tag === 'dialog' && el.open),
      box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], visible,
      inViewport: visible && r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw, obstructed,
      cursorPointer: st.cursor === 'pointer', css: cssPath(el), formId: form ? (form.id || form.getAttribute('name') || form.getAttribute('aria-label') || 'form') : null,
      legend: fieldset ? textOf(fieldset.querySelector('legend')).slice(0, 80) : null, heading,
      draggable: el.getAttribute('draggable') === 'true', editable: el.isContentEditable,
      options: tag === 'select' ? Array.from(el.options).slice(0, 200).map(o => o.text.trim()) : null,
      selectedOptions: tag === 'select' ? Array.from(el.selectedOptions).map(o => o.text.trim()) : null,
      iconHint, classes: (el.getAttribute('class') || '').split(/\s+/).slice(0, 4).join(' '),
      zIndex: st.zIndex, position: st.position,
    };
    out.push(item);
  }
  const nav = performance.getEntriesByType('navigation')[0];
  return {
    url: location.href, title: document.title, readyState: document.readyState, vw, vh,
    scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
    docHeight: Math.round(document.documentElement.scrollHeight), docWidth: Math.round(document.documentElement.scrollWidth),
    httpStatus: nav && nav.responseStatus ? nav.responseStatus : null,
    elements: out,
  };
}""")

PAGE_SIGNALS = Probe("page_signals", r"""() => {
  const textOf = (n) => (n ? (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim() : '');
  const body = document.body ? document.body.innerText : '';
  const captcha = [];
  for (const f of document.querySelectorAll('iframe')) {
    const src = f.getAttribute('src') || '';
    if (/recaptcha|hcaptcha|turnstile|challenges\.cloudflare|arkoselabs|funcaptcha|captcha/i.test(src)) captcha.push('iframe:' + src.slice(0, 80));
  }
  for (const el of document.querySelectorAll('.g-recaptcha,.h-captcha,.cf-turnstile,[class*="captcha" i],[id*="captcha" i],[data-sitekey]')) {
    captcha.push('widget:' + (el.id || el.className || el.tagName).toString().slice(0, 60));
  }
  const challengeText = body.match(/verify (that )?you are (a )?human|are you a robot|i'?m not a robot|checking your browser|unusual traffic|security check|prove you('| a)re human|complete the challenge/i);
  if (challengeText) captcha.push('text:' + challengeText[0]);
  const injection = [];
  const pattern = /(ignore|disregard|forget) (all |any )?(the |your )?(previous|prior|above|earlier) (instructions|prompts?|rules)|you are now (a|an|in) |new instructions:|system prompt|reveal (the |your )?(password|secret|api key|credentials)|(send|post|email) (the |your |all )?(credentials|password|cookies|token)|as an ai( language model)?,|override (the )?(test|expected) (result|outcome)/i;
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  let n, count = 0;
  while ((n = walker.nextNode()) && count < 20000) {
    count++;
    const t = n.nodeValue;
    if (t && t.length > 12 && pattern.test(t)) {
      const el = n.parentElement;
      const st = el ? getComputedStyle(el) : null;
      const hidden = !el || st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity || '1') < 0.05 || el.getBoundingClientRect().width === 0;
      injection.push({text: t.trim().slice(0, 160), hidden});
      if (injection.length >= 5) break;
    }
  }
  const errors = [];
  for (const el of document.querySelectorAll('[role="alert"],[aria-invalid="true"],.error,.invalid-feedback,.field-error,.error-message,[class*="error" i]')) {
    const t = textOf(el); if (t && t.length < 200 && el.getBoundingClientRect().height > 0) errors.push(t);
    if (errors.length > 10) break;
  }
  // native constraint validation (bubbles are not part of the DOM); validity is read without firing events
  const invalidFields = [];
  for (const el of document.querySelectorAll('input,select,textarea')) {
    if (el.willValidate && el.validity && !el.validity.valid && el.getBoundingClientRect().height > 0) {
      const label = (el.labels && el.labels.length ? textOf(el.labels[0]) : '') || el.getAttribute('aria-label') || el.name || el.id;
      invalidFields.push(label + ': ' + (el.validationMessage || 'invalid'));
      if (invalidFields.length > 10) break;
    }
  }
  const busy = document.querySelectorAll('[aria-busy="true"],.spinner,.loading,[class*="spinner" i],[class*="loading" i]');
  let busyVisible = 0;
  for (const b of busy) { const r = b.getBoundingClientRect(); if (r.width > 0 && r.height > 0 && getComputedStyle(b).visibility !== 'hidden') busyVisible++; }
  return {url: location.href, title: document.title, readyState: document.readyState, captcha, injection,
          validationErrors: Array.from(new Set(errors)).slice(0, 8), invalidFields, busy: busyVisible,
          textLength: body.length};
}""")

VISIBLE_TEXT = Probe("visible_text", r"""() => {
  const body = document.body ? document.body.innerText : '';
  return {url: location.href, title: document.title, text: body.slice(0, 60000)};
}""")

PAGE_INFO = Probe("page_info", r"""() => ({
  url: location.href, title: document.title, readyState: document.readyState,
  vw: window.innerWidth, vh: window.innerHeight, dpr: window.devicePixelRatio,
  scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
  docHeight: Math.round(document.documentElement.scrollHeight),
  active: document.activeElement ? (document.activeElement.getAttribute('aria-label') || document.activeElement.getAttribute('name') || document.activeElement.id || document.activeElement.tagName) : null,
})""")

ELEMENT_STATE = Probe("element_state", r"""(el) => {
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  const textOf = (n) => (n ? (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim() : '');
  const r = el.getBoundingClientRect();
  const st = getComputedStyle(el);
  return {
    tag, type, sensitive: type === 'password',
    value: type === 'password' || type === 'file' ? null : (el.value !== undefined ? String(el.value) : null),
    valueLength: el.value !== undefined && el.value !== null ? String(el.value).length : null,
    checked: (type === 'checkbox' || type === 'radio') ? !!el.checked : (el.getAttribute('aria-checked') != null ? el.getAttribute('aria-checked') === 'true' : null),
    selected: el.getAttribute('aria-selected'),
    selectedOptions: tag === 'select' ? Array.from(el.selectedOptions).map(o => o.text.trim()) : null,
    files: type === 'file' && el.files ? Array.from(el.files).map(f => f.name) : null,
    disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
    expanded: el.getAttribute('aria-expanded'), text: textOf(el).slice(0, 300),
    visible: r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none',
    box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
  };
}""", element=True)

FNV_JS = r"""const fnv = (s) => { const b = new TextEncoder().encode(s); let h = 0x811c9dc5; for (const x of b) { h ^= x; h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16); };"""

VALUE_FINGERPRINT = Probe("value_fingerprint", "(el) => { " + FNV_JS +
                          " const v = el.value !== undefined ? String(el.value) : (el.innerText || ''); "
                          "return {hash: fnv(v), length: v.length}; }", element=True)

# Renders a field that received a secret (OTP, card number, ...) as discs so no later screenshot shows the
# value. Purely presentational: the value, events and layout of the application are untouched.
MASK_SENSITIVE_FIELD = Probe("mask_sensitive_field", r"""(el) => {
  const css = window.CSS && CSS.supports && CSS.supports('-webkit-text-security', 'disc');
  if (css) { el.style.setProperty('-webkit-text-security', 'disc', 'important'); }
  else { el.style.setProperty('color', 'transparent', 'important'); el.style.setProperty('text-shadow', '0 0 0 #000', 'important'); }
  el.setAttribute('data-aitest-masked', '1');
  return {masked: true, method: css ? 'text-security' : 'transparent'};
}""", element=True, mutates_view=True)

SCROLL_INFO = Probe("scroll_info", r"""() => ({scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
  docHeight: Math.round(document.documentElement.scrollHeight), vh: window.innerHeight})""")


def fnv1a(text: str) -> str:
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return format(h, "x")


def find_text_probe(needle: str, exact: bool = False) -> Probe:
    """Locate visible elements whose own text contains ``needle`` (with boxes)."""

    literal = json.dumps(needle)
    mode = "true" if exact else "false"
    return Probe("find_text", r"""() => {
  const needle = """ + literal + r""".toLowerCase().replace(/\s+/g, ' ').trim();
  const exact = """ + mode + r""";
  const hits = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  let el;
  while ((el = walker.nextNode()) && hits.length < 25) {
    const own = Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.nodeValue).join(' ').replace(/\s+/g, ' ').trim().toLowerCase();
    const full = (el.innerText || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const match = exact ? (own === needle || full === needle) : (own.includes(needle) || (full.includes(needle) && full.length < needle.length + 60));
    if (!match) continue;
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width === 0 || r.height === 0 || st.visibility === 'hidden' || st.display === 'none') continue;
    hits.push({tag: el.tagName.toLowerCase(), text: (el.innerText || '').trim().slice(0, 120), box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]});
  }
  return {needle, hits};
}""")


def element_at_point_probe(x: float, y: float) -> Probe:
    return Probe("element_at_point", r"""() => {
  const el = document.elementFromPoint(""" + json.dumps(float(x)) + ", " + json.dumps(float(y)) + r""");
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {tag: el.tagName.toLowerCase(), id: el.id || null, role: el.getAttribute('role'), label: el.getAttribute('aria-label'),
          text: (el.innerText || '').trim().slice(0, 80), box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
          canvas: el.tagName.toLowerCase() === 'canvas'};
}""")


def scroll_probe(dx: int, dy: int) -> Probe:
    return Probe("scroll_by", "() => { window.scrollBy(" + json.dumps(int(dx)) + ", " + json.dumps(int(dy)) +
                 "); return {scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY), "
                 "docHeight: Math.round(document.documentElement.scrollHeight), vh: window.innerHeight}; }",
                 mutates_view=True)


def scroll_edge_probe(edge: str) -> Probe:
    target = "0" if edge == "top" else "document.documentElement.scrollHeight"
    return Probe("scroll_edge", "() => { window.scrollTo(0, " + target + "); return {scrollY: Math.round(window.scrollY), "
                 "docHeight: Math.round(document.documentElement.scrollHeight), vh: window.innerHeight}; }",
                 mutates_view=True)


def scroll_into_view_probe() -> Probe:
    return Probe("scroll_into_view", "(el) => { el.scrollIntoView({block: 'center', inline: 'center'}); "
                 "const r = el.getBoundingClientRect(); return {box: [Math.round(r.left), Math.round(r.top), "
                 "Math.round(r.width), Math.round(r.height)]}; }", element=True, mutates_view=True)


STATIC_PROBES: dict[str, Probe] = {p.name: p for p in (DOM_INVENTORY, PAGE_SIGNALS, VISIBLE_TEXT, PAGE_INFO,
                                                        ELEMENT_STATE, VALUE_FINGERPRINT, SCROLL_INFO,
                                                        MASK_SENSITIVE_FIELD)}
PROBE_NAMES = set(STATIC_PROBES) | {"find_text", "element_at_point", "scroll_by", "scroll_edge", "scroll_into_view"}
