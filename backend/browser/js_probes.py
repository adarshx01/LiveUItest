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


_DEEP_JS = r"""
// every element, including the content of open shadow roots (web components), in document order
const deepElements = (root, limit) => {
  const out = [];
  const stack = [root];
  while (stack.length && out.length < limit) {
    const node = stack.pop();
    const kids = node.children ? Array.from(node.children) : [];
    if (node.shadowRoot && node.shadowRoot.children) kids.push(...Array.from(node.shadowRoot.children));
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]);
    if (node.nodeType === 1 && node !== root) out.push(node);
  }
  return out;
};
const rootHostOf = (el) => { const r = el.getRootNode && el.getRootNode(); return r && r.host ? r.host : null; };
const deepText = (root) => {
  let text = root.innerText || '';
  for (const el of deepElements(root, 20000)) {
    if (el.shadowRoot) for (const c of el.shadowRoot.children) {
      if (/^(STYLE|SCRIPT|TEMPLATE|LINK)$/.test(c.tagName)) continue;
      const t = c.innerText || ''; if (t) text += '\n' + t;
    }
  }
  return text;
};
"""

_CSS_PATH_JS = r"""
const cssEscape = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
const looksGenerated = (s) => !s || /\d{4,}|[a-f0-9]{8,}|^(ember|react|mui|css|sc|jsx|ng|_)[-_]?|^[a-z]{1,3}[0-9a-z]{4,}$/i.test(s) && /\d/.test(s);
const scopeOf = (el) => el.getRootNode ? el.getRootNode() : document;
const uniq = (sel, scope) => { try { return (scope || document).querySelectorAll(sel).length === 1; } catch (e) { return false; } };
let uniqBudget = 8000;
const localPath = (el) => {
  const scope = scopeOf(el);
  const tid = el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa');
  if (tid) { const attr = el.getAttribute('data-testid') ? 'data-testid' : (el.getAttribute('data-test') ? 'data-test' : 'data-qa');
             const s = `[${attr}="${tid}"]`; if (uniq(s, scope)) return s; }
  if (el.id && !looksGenerated(el.id)) { const s = '#' + cssEscape(el.id); if (uniq(s, scope)) return s; }
  const nm = el.getAttribute('name');
  if (nm) { const s = `${el.tagName.toLowerCase()}[name="${nm}"]`; if (uniq(s, scope)) return s; }
  const parts = [];
  let cur = el;
  for (let depth = 0; cur && cur.nodeType === 1 && depth < 12; depth++) {
    let part = cur.tagName.toLowerCase();
    if (cur.id && !looksGenerated(cur.id)) { parts.unshift('#' + cssEscape(cur.id)); break; }
    const parent = cur.parentElement || (cur.parentNode && cur.parentNode.host ? cur.parentNode : null);
    if (parent) {
      const same = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
      if (same.length > 1) part += `:nth-of-type(${same.indexOf(cur) + 1})`;
    }
    parts.unshift(part);
    if (depth >= 2 && uniqBudget-- > 0 && uniq(parts.join(' > '), scope)) break;  // shortest unique path
    cur = cur.parentElement;
  }
  return parts.join(' > ');
};
// Playwright CSS pierces open shadow roots, so "host-path inner-path" addresses web-component internals
const cssPath = (el) => {
  if (!(el instanceof Element)) return null;
  const inner = localPath(el);
  const host = rootHostOf(el);
  return host ? `${cssPath(host)} ${inner}` : inner;
};
"""

_LABEL_JS = r"""
const textOf = (n) => (n ? (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim() : '');
const byId = (el, id) => { const r = el.getRootNode ? el.getRootNode() : document; return (r.getElementById ? r.getElementById(id) : null) || document.getElementById(id); };
const isControl = (n) => n && n.nodeType === 1 && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(n.tagName);
const shortText = (t) => { t = (t || '').replace(/\s+/g, ' ').replace(/[:*]\s*$/, '').trim(); return t.length > 0 && t.length <= 60 ? t : ''; };
const labelFor = (el) => {
  const by = el.getAttribute('aria-labelledby');
  if (by) { const t = by.split(/\s+/).map(id => textOf(byId(el, id))).join(' ').trim(); if (t) return t; }
  if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
  if (el.labels && el.labels.length) return textOf(el.labels[0]);
  const wrap = el.closest('label'); if (wrap) return textOf(wrap);
  return '';
};
// Messy pages rarely associate labels: read the text a human would read as the field's label.
const ownText = (n) => Array.from(n.childNodes).filter(c => c.nodeType === 3).map(c => c.nodeValue).join(' ');
const inferLabel = (el) => {
  // 1. text right before the control inside the same container ("Email: <input>", "<span>Email</span><input>")
  for (let sib = el.previousSibling, hops = 0; sib && hops < 4; sib = sib.previousSibling, hops++) {
    if (sib.nodeType === 3) { const t = shortText(sib.nodeValue); if (t) return t; continue; }
    if (sib.nodeType !== 1) continue;
    if (isControl(sib) || sib.querySelector && sib.querySelector('input,select,textarea,button')) break;
    const t = shortText(textOf(sib)); if (t) return t;
  }
  // 2. table forms: the cell to the left, else the column header
  const cell = el.closest('td,th,[role="gridcell"],[role="cell"]');
  if (cell) {
    for (let prev = cell.previousElementSibling; prev; prev = prev.previousElementSibling) {
      const t = shortText(textOf(prev)); if (t && !prev.querySelector('input,select,textarea')) return t;
    }
    const table = cell.closest('table');
    if (table && cell.parentElement) {
      const col = Array.from(cell.parentElement.children).indexOf(cell);
      const head = table.querySelector('thead tr, tr');
      if (head && head !== cell.parentElement && head.children[col]) { const t = shortText(textOf(head.children[col])); if (t) return t; }
    }
  }
  // 3. a wrapper whose own text is the label ("<div>Email <input></div>"), or the wrapper's previous sibling
  let wrapper = el.parentElement;
  for (let depth = 0; wrapper && depth < 3; depth++, wrapper = wrapper.parentElement) {
    const controls = wrapper.querySelectorAll('input,select,textarea');
    if (controls.length > 1) break;
    const own = shortText(ownText(wrapper)); if (own) return own;
    const prev = wrapper.previousElementSibling;
    if (prev && !isControl(prev) && !(prev.querySelector && prev.querySelector('input,select,textarea,button'))) {
      const t = shortText(textOf(prev)); if (t) return t;
    }
  }
  // 4. definition lists
  const dd = el.closest('dd'); if (dd && dd.previousElementSibling && dd.previousElementSibling.tagName === 'DT') return shortText(textOf(dd.previousElementSibling));
  return '';
};
const describedBy = (el) => { const by = el.getAttribute('aria-describedby'); return by ? by.split(/\s+/).map(id => textOf(byId(el, id))).join(' ').trim().slice(0, 120) : ''; };
"""

DOM_INVENTORY = Probe("dom_inventory", r"""() => {
""" + _DEEP_JS + _CSS_PATH_JS + _LABEL_JS + r"""
  const MAX = 800;
  const vw = window.innerWidth, vh = window.innerHeight;
  const sel = 'a[href],button,input,select,textarea,summary,details,[role],[tabindex],[onclick],[contenteditable="true"],' +
              '[contenteditable=""],canvas,svg[aria-label],img,[data-testid],[draggable="true"],iframe,label,h1,h2,h3,h4,form,dialog,[aria-modal="true"]';
  const POINTER_TAGS = new Set(['DIV', 'SPAN', 'LI', 'I', 'SVG', 'TD', 'P', 'A', 'IMG', 'B', 'STRONG', 'EM', 'SECTION',
                                'ARTICLE', 'TR', 'H5', 'H6', 'FIGURE', 'LABEL', 'UL', 'DT', 'DD']);
  const all = deepElements(document.body || document.documentElement, 15000);
  const nodes = [];
  let scanned = 0;
  for (const el of all) {
    let take = false;
    try { take = el.matches(sel); } catch (e) { take = false; }
    if (!take && POINTER_TAGS.has(el.tagName.toUpperCase()) && scanned++ < 5000) {
      // clickable-looking elements without semantics (div soup): pointer cursor not inherited from a parent
      const st = getComputedStyle(el);
      const parent = el.parentElement || rootHostOf(el);
      if (st.cursor === 'pointer' && !(parent && getComputedStyle(parent).cursor === 'pointer')) take = true;
      else if (el.onclick || el.hasAttribute('onmousedown') || el.hasAttribute('ng-click') || el.hasAttribute('@click') || el.hasAttribute('v-on:click')) take = true;
    }
    if (take) nodes.push(el);
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
        let top = document.elementFromPoint(cx, cy);
        while (top && top.shadowRoot) { const inner = top.shadowRoot.elementFromPoint(cx, cy); if (!inner || inner === top) break; top = inner; }
        const host = rootHostOf(el);
        obstructed = !!top && top !== el && !el.contains(top) && !top.contains(el) && !(host && (host === top || host.contains(top)))
                     && !(el.labels && Array.from(el.labels).some(l => l.contains(top)));
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
    const isField = tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
    const label = labelFor(el);
    const item = {
      tag, type, id: el.id || null, name: el.getAttribute('name'), testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa'),
      role: el.getAttribute('role'), ariaLabel: el.getAttribute('aria-label'), title: el.getAttribute('title'),
      placeholder: el.getAttribute('placeholder'), autocomplete: el.getAttribute('autocomplete'), label: label.slice(0, 120),
      inferredLabel: !label && (isField || /^(combobox|textbox|listbox)$/.test(el.getAttribute('role') || '') || el.getAttribute('aria-haspopup') ||
                     /(^|[\s_-])(select|dropdown|dd|picker|combo)([\s_-]|$)/i.test(typeof el.className === 'string' ? el.className : '') ||
                     /^(select|choose|pick)\b/i.test(textOf(el))) ? inferLabel(el).slice(0, 80) : '',
      description: describedBy(el),
      text: textOf(el).slice(0, 120), href: el.getAttribute('href'), alt: el.getAttribute('alt'),
      value: (tag === 'input' || tag === 'textarea' || tag === 'select') && type !== 'password' && type !== 'file' ? String(el.value || '').slice(0, 120) : null,
      valueLength: (tag === 'input' || tag === 'textarea') ? String(el.value || '').length : null,
      checked: (type === 'checkbox' || type === 'radio') ? !!el.checked : (el.getAttribute('aria-checked') != null ? el.getAttribute('aria-checked') === 'true' : null),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true', readonly: !!el.readOnly, required: !!el.required || el.getAttribute('aria-required') === 'true',
      expanded: el.getAttribute('aria-expanded'), modal: el.getAttribute('aria-modal') === 'true' || (tag === 'dialog' && el.open),
      box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], visible,
      inViewport: visible && r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw, obstructed,
      cursorPointer: st.cursor === 'pointer', css: cssPath(el), shadow: !!rootHostOf(el),
      formId: form ? (form.id || form.getAttribute('name') || form.getAttribute('aria-label') || 'form') : null,
      legend: fieldset ? textOf(fieldset.querySelector('legend')).slice(0, 80) : null, heading,
      draggable: el.getAttribute('draggable') === 'true', editable: el.isContentEditable,
      options: tag === 'select' ? Array.from(el.options).slice(0, 300).map(o => (o.label || o.text || '').trim()) : null,
      optionDetails: tag === 'select' ? Array.from(el.options).slice(0, 300).map((o, i) => ({label: (o.label || o.text || '').trim(), value: o.value,
                       disabled: o.disabled || !!(o.parentElement && o.parentElement.tagName === 'OPTGROUP' && o.parentElement.disabled),
                       group: o.parentElement && o.parentElement.tagName === 'OPTGROUP' ? o.parentElement.label : null, selected: o.selected, index: i})) : null,
      selectedOptions: tag === 'select' ? Array.from(el.selectedOptions).map(o => (o.label || o.text || '').trim()) : null,
      multiple: tag === 'select' ? !!el.multiple : el.getAttribute('aria-multiselectable') === 'true',
      datalist: tag === 'input' && el.list ? Array.from(el.list.options).slice(0, 300).map(o => (o.value || o.label || o.text || '').trim()) : null,
      min: el.getAttribute('min'), max: el.getAttribute('max'), step: el.getAttribute('step'),
      controls: el.getAttribute('aria-controls') || el.getAttribute('aria-owns') || null, haspopup: el.getAttribute('aria-haspopup'),
      activeDescendant: el.getAttribute('aria-activedescendant'),
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
    shadowHosts: all.filter(e => e.shadowRoot).length,
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
""" + _DEEP_JS + r"""
  const body = document.body ? deepText(document.body) : '';
  return {url: location.href, title: document.title, text: body.slice(0, 80000)};
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


# Live state of a native <select> (also accepts an <option> or a wrapper around the select). Option
# lists of dependent selects are filled asynchronously, so the executor reads them at action time.
SELECT_STATE = Probe("select_state", r"""(el) => {
  let sel = el;
  if (!sel || sel.tagName !== 'SELECT') sel = (el.closest && el.closest('select')) || (el.querySelector && el.querySelector('select'));
  if (!sel || sel.tagName !== 'SELECT') return {isSelect: false};
  const r = sel.getBoundingClientRect(); const st = getComputedStyle(sel);
  const options = Array.from(sel.options).slice(0, 500).map((o, i) => ({
    label: (o.label || o.text || '').trim(), value: o.value,
    disabled: o.disabled || !!(o.parentElement && o.parentElement.tagName === 'OPTGROUP' && o.parentElement.disabled),
    group: o.parentElement && o.parentElement.tagName === 'OPTGROUP' ? o.parentElement.label : null, selected: o.selected, index: i}));
  return {isSelect: true, multiple: sel.multiple, disabled: sel.disabled, size: sel.size,
          visible: r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none',
          options, selected: options.filter(o => o.selected).map(o => o.label), value: sel.value};
}""", element=True)

# Input constraints and <datalist> suggestions (rendered by the browser, not part of the page's DOM UI).
INPUT_STATE = Probe("input_state", r"""(el) => {
  const type = (el.getAttribute('type') || (el.tagName === 'TEXTAREA' ? 'textarea' : 'text')).toLowerCase();
  const list = el.list ? Array.from(el.list.options).slice(0, 300).map(o => ({label: (o.label || o.value || o.text || '').trim(), value: o.value})) : [];
  return {tag: el.tagName.toLowerCase(), type, min: el.getAttribute('min'), max: el.getAttribute('max'), step: el.getAttribute('step'),
          value: type === 'password' ? null : (el.value !== undefined ? String(el.value) : null), datalist: list,
          readonly: !!el.readOnly, disabled: !!el.disabled, editable: !!el.isContentEditable};
}""", element=True)

# Options of the popup a trigger opened: the listbox it controls (aria-controls/aria-owns), otherwise the
# visible list-like containers closest to it (portals at the end of <body> included), otherwise ARIA-less
# "div soup" menus. Returned with unique selectors and boxes so they can be clicked without refs.
POPUP_OPTIONS = Probe("popup_options", r"""(trigger) => {
""" + _DEEP_JS + _CSS_PATH_JS + _LABEL_JS + r"""
  const visible = (e) => { const r = e.getBoundingClientRect(); const st = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none' && parseFloat(st.opacity || '1') > 0.02; };
  const tr = trigger.getBoundingClientRect();
  const OPT = '[role="option"],[role="menuitem"],[role="menuitemradio"],[role="menuitemcheckbox"],[role="treeitem"]';
  const idsOf = (e) => (e.getAttribute('aria-controls') || e.getAttribute('aria-owns') || '').split(/\s+/).filter(Boolean);
  let source = 'aria-controls';
  let containers = idsOf(trigger).map(id => byId(trigger, id)).filter(c => c && visible(c));
  if (!containers.length) {
    const wrap = trigger.closest('[aria-controls],[aria-owns]');
    if (wrap) containers = idsOf(wrap).map(id => byId(wrap, id)).filter(c => c && visible(c));
  }
  const all = deepElements(document.body, 15000);
  if (!containers.length) {
    source = 'nearby-popup';
    const cls = (e) => (typeof e.className === 'string' ? e.className : '');
    containers = all.filter(e => e !== trigger && !e.contains(trigger) && visible(e) &&
      (/^(listbox|menu|tree|grid)$/.test(e.getAttribute('role') || '') ||
       /(^|[\s_-])(menu|dropdown|dd-menu|options|option-list|listbox|popover|popup|select-?list|suggestions?|results|choices)([\s_-]|$)/i.test(cls(e))));
    // innermost containers only
    containers = containers.filter(c => !containers.some(o => o !== c && c.contains(o) && o.querySelector(OPT + ',li,[data-value]')));
  }
  const dist = (c) => { const r = c.getBoundingClientRect(); return Math.abs(r.top - tr.bottom) + Math.abs(r.left - tr.left) * 0.5; };
  containers.sort((a, b) => dist(a) - dist(b));
  const optionsIn = (c) => {
    let opts = Array.from(c.querySelectorAll(OPT));
    if (!opts.length) opts = Array.from(c.querySelectorAll('li,[data-value],[data-option],a,button,div,span'))
      .filter(e => textOf(e) && !e.querySelector('li,[data-value],[data-option],a,button') && e.children.length <= 3 && textOf(e).length < 120);
    return opts.filter(visible);
  };
  const out = [];
  for (const c of containers.slice(0, 3)) {
    for (const o of optionsIn(c)) {
      if (out.length >= 250) break;
      const r = o.getBoundingClientRect();
      const cls = typeof o.className === 'string' ? o.className : '';
      out.push({text: textOf(o).slice(0, 120), value: o.getAttribute('data-value') || o.getAttribute('value'), role: o.getAttribute('role'),
                selected: o.getAttribute('aria-selected') === 'true' || o.getAttribute('aria-checked') === 'true' || /(^|\s)(selected|is-selected|active-option|checked)(\s|$)/.test(cls),
                disabled: o.getAttribute('aria-disabled') === 'true' || o.hasAttribute('disabled') || /(^|\s)(disabled|is-disabled)(\s|$)/.test(cls),
                haspopup: o.getAttribute('aria-haspopup'), css: cssPath(o), box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]});
    }
  }
  let search = null;
  for (const c of containers.slice(0, 1)) {
    const scope = [c, c.parentElement].filter(Boolean);
    for (const s of scope) {
      const box = s.querySelector('input[type="search"],input[type="text"],input:not([type]),[role="searchbox"],[role="textbox"]');
      if (box && box !== trigger && visible(box)) { search = cssPath(box); break; }
    }
  }
  const scrollable = containers.length ? (() => { let c = containers[0]; for (let i = 0; c && i < 4; i++, c = c.parentElement) {
      const st = getComputedStyle(c); if (/(auto|scroll)/.test(st.overflowY) && c.scrollHeight > c.clientHeight + 2) return true; } return false; })() : false;
  return {source: containers.length ? source : 'none', containers: containers.length, scrollable, search,
          expanded: trigger.getAttribute('aria-expanded'), options: out};
}""", element=True)

# Scroll the list that contains ``el`` (virtualized listboxes render only the rows in view).
SCROLL_LIST = Probe("scroll_list", r"""(el) => {
  let c = el;
  for (let i = 0; c && c !== document.body && i < 8; i++, c = c.parentElement || (c.parentNode && c.parentNode.host)) {
    const st = getComputedStyle(c);
    if (/(auto|scroll)/.test(st.overflowY) && c.scrollHeight > c.clientHeight + 2) break;
  }
  if (!c || c === document.body) return {scrolled: false, reason: 'no scrollable list'};
  const before = c.scrollTop;
  c.scrollTop = Math.min(c.scrollTop + Math.max(40, c.clientHeight * 0.85), c.scrollHeight);
  c.dispatchEvent(new Event('scroll'));
  return {scrolled: c.scrollTop !== before, atEnd: c.scrollTop + c.clientHeight >= c.scrollHeight - 2,
          top: Math.round(c.scrollTop), height: c.scrollHeight, client: c.clientHeight};
}""", element=True, mutates_view=True)


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
""" + _DEEP_JS + r"""
  for (const el of deepElements(document.body, 20000)) {
    if (hits.length >= 25) break;
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
                                                        MASK_SENSITIVE_FIELD, SELECT_STATE, INPUT_STATE,
                                                        POPUP_OPTIONS, SCROLL_LIST)}
PROBE_NAMES = set(STATIC_PROBES) | {"find_text", "element_at_point", "scroll_by", "scroll_edge", "scroll_into_view"}
