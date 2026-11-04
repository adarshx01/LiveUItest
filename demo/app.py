"""Controlled demo application used to test the testing platform (spec section 29).

Every page includes a ground-truth recorder: interactions with elements that
carry ``data-gt`` attributes are reported to ``/__events`` so evaluations can
measure *which* element the agent actually acted on, not just whether the test
appeared to pass.

Run: ``python -m demo.app`` (port 8800)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
                        autoescape=select_autoescape(["html"]))

USERS = {
    "abc": {"password": "abc", "display": "abc"},
    "test@example.com": {"password": "Passw0rd!", "display": "test@example.com"},
    "alice@example.com": {"password": "Alice#2024", "display": "Alice Smith"},
    "bob": {"password": "bobpass", "display": "bob"},
}

PRODUCTS = [
    {"id": i + 1, "name": name, "category": cat, "price": price}
    for i, (name, cat, price) in enumerate([
        ("Laptop Pro 14", "Computers", 1899), ("Laptop Air 13", "Computers", 1199), ("Desktop Tower", "Computers", 1499),
        ("Wireless Mouse", "Accessories", 39), ("Mechanical Keyboard", "Accessories", 129), ("USB-C Hub", "Accessories", 59),
        ("27in Monitor", "Displays", 349), ("34in Ultrawide", "Displays", 699), ("Portable Monitor", "Displays", 229),
        ("Noise Cancelling Headphones", "Audio", 299), ("Bluetooth Speaker", "Audio", 89), ("Studio Microphone", "Audio", 149),
        ("Webcam 4K", "Accessories", 119), ("Laptop Stand", "Accessories", 49), ("Docking Station", "Accessories", 229),
        ("Gaming Laptop", "Computers", 2299), ("Tablet 11", "Computers", 799), ("E-Reader", "Computers", 139),
        ("Soundbar", "Audio", 399), ("Earbuds", "Audio", 179), ("OLED TV 55in", "Displays", 1499),
        ("Projector", "Displays", 899), ("Graphics Tablet", "Accessories", 259), ("Smart Watch", "Accessories", 349),
    ])
]

STATES = {
    "India": ["Kerala", "Karnataka", "Tamil Nadu", "Maharashtra", "Delhi"],
    "United States": ["California", "Texas", "New York", "Florida", "Washington"],
    "Canada": ["Ontario", "Quebec", "British Columbia"],
}
CITIES = {
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Delhi": ["New Delhi"],
    "California": ["Los Angeles", "San Francisco", "San Diego"],
    "Texas": ["Austin", "Houston", "Dallas"],
    "New York": ["New York City", "Buffalo"],
    "Florida": ["Miami", "Orlando"],
    "Washington": ["Seattle", "Spokane"],
    "Ontario": ["Toronto", "Ottawa"],
    "Quebec": ["Montreal", "Quebec City"],
    "British Columbia": ["Vancouver", "Victoria"],
}
COUNTRIES = ["Australia", "Brazil", "Canada", "France", "Germany", "India", "Ireland", "Japan", "Mexico",
             "Netherlands", "New Zealand", "Singapore", "South Africa", "Spain", "Sweden", "United Kingdom",
             "United States"]


class DemoState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ui_version = 1
        self.challenge_enabled = False
        self.events: list[dict[str, Any]] = []
        self.sessions: dict[str, str] = {}
        self.carts: dict[str, list[int]] = {}
        self.slow_ms = 0
        self.checkout_broken = True
        self.care_sessions: dict[str, str] = {}
        self.care_logs: dict[int, list[dict[str, Any]]] = {}


STATE = DemoState()
app = FastAPI(title="Demo application under test", docs_url=None, redoc_url=None)


def render(name: str, request: Request, **ctx: Any) -> HTMLResponse:
    user = current_user(request)
    html = TEMPLATES.get_template(name).render(request=request, user=user, ui_version=STATE.ui_version,
                                               cart_count=len(STATE.carts.get(request.cookies.get("sid", ""), [])),
                                               **ctx)
    return HTMLResponse(html)


def current_user(request: Request) -> str | None:
    sid = request.cookies.get("sid")
    return STATE.sessions.get(sid) if sid else None


# --------------------------------------------------------------------------- admin / ground truth
@app.post("/__events")
async def record_event(request: Request) -> Response:
    try:
        payload = json.loads((await request.body()) or b"{}")
    except json.JSONDecodeError:
        payload = {}
    payload["ts"] = time.time()
    STATE.events.append(payload)
    return Response(status_code=204)


@app.get("/__events")
async def list_events() -> JSONResponse:
    return JSONResponse(STATE.events)


@app.delete("/__events")
async def clear_events() -> Response:
    STATE.events.clear()
    return Response(status_code=204)


@app.post("/__admin/reset")
async def admin_reset() -> JSONResponse:
    STATE.reset()
    return JSONResponse({"ok": True})


@app.post("/__admin/config")
async def admin_config(request: Request) -> JSONResponse:
    data = await request.json()
    if "ui_version" in data:
        STATE.ui_version = int(data["ui_version"])
    if "challenge_enabled" in data:
        STATE.challenge_enabled = bool(data["challenge_enabled"])
    if "slow_ms" in data:
        STATE.slow_ms = int(data["slow_ms"])
    if "checkout_broken" in data:
        STATE.checkout_broken = bool(data["checkout_broken"])
    return JSONResponse({"ui_version": STATE.ui_version, "challenge_enabled": STATE.challenge_enabled,
                         "slow_ms": STATE.slow_ms, "checkout_broken": STATE.checkout_broken})


@app.get("/__health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- core pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return render("home.html", request, show_cookies=request.query_params.get("cookies") == "1")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Response:
    if STATE.challenge_enabled and request.cookies.get("human") != "1":
        return RedirectResponse("/challenge?next=/login", status_code=303)
    template = "login_v2.html" if STATE.ui_version == 2 else "login.html"
    return render(template, request, error=None, identifier="")


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, identifier: str = Form(""), password: str = Form(""),
                       remember: str | None = Form(None)) -> Response:
    if STATE.slow_ms:
        await asyncio.sleep(STATE.slow_ms / 1000)
    template = "login_v2.html" if STATE.ui_version == 2 else "login.html"
    if not identifier or not password:
        return render(template, request, error="Please enter your username and password", identifier=identifier)
    user = USERS.get(identifier.strip())
    if user is None or user["password"] != password:
        return render(template, request, error="Invalid username or password", identifier=identifier)
    sid = f"s{len(STATE.sessions) + 1}{int(time.time() * 1000)}"
    STATE.sessions[sid] = identifier.strip()
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("sid", sid, httponly=True, samesite="lax")
    return response


@app.get("/logout")
async def logout(request: Request) -> Response:
    sid = request.cookies.get("sid")
    STATE.sessions.pop(sid or "", None)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("sid")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> Response:
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    display = USERS.get(user, {}).get("display", user)
    return render("dashboard.html", request, display_name=display)


@app.get("/challenge", response_class=HTMLResponse)
async def challenge(request: Request) -> HTMLResponse:
    return render("challenge.html", request, next_url=request.query_params.get("next", "/"))


@app.post("/challenge/verify")
async def challenge_verify(request: Request, next_url: str = Form("/")) -> Response:
    response = RedirectResponse(next_url if next_url.startswith("/") else "/", status_code=303)
    response.set_cookie("human", "1", samesite="lax")
    return response


@app.get("/controls", response_class=HTMLResponse)
async def controls_page(request: Request) -> HTMLResponse:
    return render("controls.html", request)


@app.get("/dropdowns", response_class=HTMLResponse)
async def dropdowns_page(request: Request) -> HTMLResponse:
    return render("dropdowns.html", request)


@app.get("/soup", response_class=HTMLResponse)
async def soup_page(request: Request) -> HTMLResponse:
    return render("soup.html", request)


OTP_CODE = "246810"


@app.get("/verify-code", response_class=HTMLResponse)
async def verify_code_page(request: Request) -> HTMLResponse:
    return render("verify_code.html", request, accepted=False, error=None)


@app.post("/verify-code", response_class=HTMLResponse)
async def verify_code(request: Request, code: str = Form("")) -> HTMLResponse:
    if code.strip() != OTP_CODE:
        return render("verify_code.html", request, accepted=False, error="That code is not valid")
    return render("verify_code.html", request, accepted=True, error=None)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    return render("register.html", request, countries=COUNTRIES, errors={}, values={})


@app.post("/register", response_class=HTMLResponse)
async def register_submit(request: Request) -> HTMLResponse:
    form = await request.form()
    values = {k: str(v) for k, v in form.items()}
    errors: dict[str, str] = {}
    if not values.get("full_name"):
        errors["full_name"] = "Full name is required"
    email = values.get("email", "")
    if not email:
        errors["email"] = "Email is required"
    elif "@" not in email or "." not in email.split("@")[-1]:
        errors["email"] = "Enter a valid email address"
    if len(values.get("password", "")) < 8:
        errors["password"] = "Password must be at least 8 characters"
    if values.get("password") != values.get("confirm_password"):
        errors["confirm_password"] = "Passwords do not match"
    if values.get("account_type") == "Business" and not values.get("company"):
        errors["company"] = "Company name is required for business accounts"
    if not values.get("terms"):
        errors["terms"] = "You must accept the terms and conditions"
    if errors:
        return render("register.html", request, countries=COUNTRIES, errors=errors, values=values)
    return render("register_success.html", request, values=values)


@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request) -> HTMLResponse:
    return render("profile.html", request, countries=list(STATES.keys()) + ["Germany", "Japan"],
                  states_json=json.dumps(STATES), cities_json=json.dumps(CITIES))


@app.post("/profile", response_class=HTMLResponse)
async def profile_submit(request: Request) -> HTMLResponse:
    form = await request.form()
    values: dict[str, Any] = {}
    for key in form.keys():
        items = form.getlist(key)
        values[key] = ", ".join(str(i) for i in items)
    return render("profile_success.html", request, values=values)


@app.get("/api/cities")
async def api_cities(state: str, q: str = "") -> JSONResponse:
    await asyncio.sleep(0.2)
    cities = [c for c in CITIES.get(state, []) if q.lower() in c.lower()]
    return JSONResponse(cities)


@app.get("/api/countries")
async def api_countries(q: str = "") -> JSONResponse:
    await asyncio.sleep(0.15)
    return JSONResponse([c for c in COUNTRIES if c.lower().startswith(q.lower())][:8])


@app.get("/autocomplete", response_class=HTMLResponse)
async def autocomplete_page(request: Request) -> HTMLResponse:
    return render("autocomplete.html", request)


# --------------------------------------------------------------------------- shop / checkout
@app.get("/shop", response_class=HTMLResponse)
async def shop(request: Request) -> HTMLResponse:
    q = request.query_params.get("q", "").strip()
    category = request.query_params.get("category", "")
    page = max(1, int(request.query_params.get("page", "1") or 1))
    items = [p for p in PRODUCTS if (not q or q.lower() in p["name"].lower())
             and (not category or p["category"] == category)]
    per_page = 8
    pages = max(1, (len(items) + per_page - 1) // per_page)
    page = min(page, pages)
    shown = items[(page - 1) * per_page: page * per_page]
    return render("shop.html", request, products=shown, q=q, category=category, page=page, pages=pages,
                  total=len(items), categories=sorted({p["category"] for p in PRODUCTS}))


@app.post("/cart/add")
async def cart_add(request: Request, product_id: int = Form(...)) -> Response:
    sid = request.cookies.get("sid") or f"anon{int(time.time() * 1000)}"
    STATE.carts.setdefault(sid, []).append(product_id)
    response = RedirectResponse(request.headers.get("referer") or "/shop", status_code=303)
    response.set_cookie("sid", sid, samesite="lax")
    return response


@app.get("/cart", response_class=HTMLResponse)
async def cart(request: Request) -> HTMLResponse:
    ids = STATE.carts.get(request.cookies.get("sid", ""), [])
    items = [p for pid in ids for p in PRODUCTS if p["id"] == pid]
    return render("cart.html", request, items=items, total=sum(p["price"] for p in items))


@app.post("/checkout", response_class=HTMLResponse)
async def checkout(request: Request) -> Response:
    ids = STATE.carts.get(request.cookies.get("sid", ""), [])
    if STATE.checkout_broken:
        # deliberately broken workflow: the server fails and the UI shows a generic error
        response = render("checkout_error.html", request)
        response.status_code = 500
        return response
    total = sum(p["price"] for pid in ids for p in PRODUCTS if p["id"] == pid)
    return render("checkout_success.html", request, total=total, order_id=f"ORD-{1000 + len(ids)}")


# --------------------------------------------------------------------------- widgets
@app.get("/widgets", response_class=HTMLResponse)
async def widgets(request: Request) -> HTMLResponse:
    return render("widgets.html", request)


@app.get("/widgets/{name}", response_class=HTMLResponse)
async def widget(request: Request, name: str) -> Response:
    allowed = {"modal", "tabs", "iframe", "upload", "dynamic", "icons", "dnd", "canvas", "infinite", "responsive",
               "slow", "injection"}
    if name not in allowed:
        return HTMLResponse("Not found", status_code=404)
    return render(f"w_{name}.html", request)


@app.get("/slow-result", response_class=HTMLResponse)
async def slow_result(request: Request) -> HTMLResponse:
    await asyncio.sleep(2.0)
    return render("slow_result.html", request)


@app.get("/terms", response_class=HTMLResponse)
async def terms(request: Request) -> HTMLResponse:
    return render("terms.html", request)


@app.get("/frames/newsletter", response_class=HTMLResponse)
async def frame_newsletter(request: Request) -> HTMLResponse:
    return render("frame_newsletter.html", request)


@app.get("/files/report.csv")
async def report_csv() -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["product", "category", "price"])
    for p in PRODUCTS[:10]:
        writer.writerow([p["name"], p["category"], p["price"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=report.csv"})


@app.get("/api/items")
async def api_items(offset: int = 0, limit: int = 20) -> JSONResponse:
    await asyncio.sleep(0.3)
    total = 100
    return JSONResponse({"items": [f"Item {i}" for i in range(offset + 1, min(total, offset + limit) + 1)],
                         "total": total})


# ---------------------------------------------------------------------------------------------------------------
# /care: a patient-services portal shaped like the programme/interaction screens of a healthcare CRM. Sign-in,
# programme list, patient table with near-duplicate names, tabs, and a "Record Interaction" drawer with three
# kinds of custom dropdown, a calendar date picker and an HH / MM time popover. Interactions persist server-side.
# The demo account is care.user@example.com / CareDemo#2026; DEMO_CARE_USERS ("email:password" pairs separated
# by ";") adds more at start-up. Never put real credentials in code.

PATIENTS = [
    {"id": 101, "name": "Abc Xyz", "code": "PSP-00101", "enrolled": "12 Jan 2026", "status": "Active"},
    {"id": 102, "name": "Abc Xy", "code": "PSP-00102", "enrolled": "03 Feb 2026", "status": "Active"},
    {"id": 103, "name": "Bcd Ef", "code": "PSP-00103", "enrolled": "21 Mar 2026", "status": "On hold"},
    {"id": 104, "name": "Abcd Xy", "code": "PSP-00104", "enrolled": "07 Apr 2026", "status": "Active"},
]


def care_users() -> dict[str, str]:
    import os

    users: dict[str, str] = {"care.user@example.com": "CareDemo#2026"}  # demo account, always present
    for pair in os.environ.get("DEMO_CARE_USERS", "").split(";"):
        email, sep, password = pair.strip().partition(":")
        if sep and email:
            users[email.strip().lower()] = password
    return users


def care_user(request: Request) -> str | None:
    sid = request.cookies.get("care_sid")
    return STATE.care_sessions.get(sid) if sid else None


def care_render(name: str, request: Request, **ctx: Any) -> HTMLResponse:
    html = TEMPLATES.get_template(f"care/{name}").render(request=request, care_user=care_user(request), **ctx)
    return HTMLResponse(html)


@app.get("/care", response_class=HTMLResponse)
@app.get("/care/sign-in", response_class=HTMLResponse)
async def care_sign_in(request: Request) -> Response:
    if care_user(request):
        return RedirectResponse("/care/home", status_code=303)
    return care_render("signin.html", request, error=None, email="")


@app.post("/care/sign-in", response_class=HTMLResponse)
async def care_sign_in_submit(request: Request, email: str = Form(""), password: str = Form("")) -> Response:
    await asyncio.sleep(0.4)  # a real authentication round trip
    if care_users().get(email.strip().lower()) != password or not password:
        return care_render("signin.html", request, error="Invalid email or password", email=email)
    sid = f"c{len(STATE.care_sessions) + 1}{int(time.time() * 1000)}"
    STATE.care_sessions[sid] = email.strip()
    response = RedirectResponse("/care/home", status_code=303)
    response.set_cookie("care_sid", sid, httponly=True, samesite="lax")
    return response


@app.get("/care/logout")
async def care_logout(request: Request) -> Response:
    STATE.care_sessions.pop(request.cookies.get("care_sid") or "", None)
    response = RedirectResponse("/care/sign-in", status_code=303)
    response.delete_cookie("care_sid")
    return response


@app.get("/care/home", response_class=HTMLResponse)
async def care_home(request: Request) -> Response:
    if not care_user(request):
        return RedirectResponse("/care/sign-in", status_code=303)
    return care_render("home.html", request)


@app.get("/care/programs/psp", response_class=HTMLResponse)
async def care_psp(request: Request) -> Response:
    if not care_user(request):
        return RedirectResponse("/care/sign-in", status_code=303)
    return care_render("psp.html", request, patients=PATIENTS)


@app.get("/care/patients/{patient_id}", response_class=HTMLResponse)
async def care_patient(request: Request, patient_id: int) -> Response:
    if not care_user(request):
        return RedirectResponse("/care/sign-in", status_code=303)
    patient = next((p for p in PATIENTS if p["id"] == patient_id), None)
    if patient is None:
        return HTMLResponse("patient not found", status_code=404)
    return care_render("patient.html", request, patient=patient, logs=STATE.care_logs.get(patient_id, []))


@app.post("/care/api/patients/{patient_id}/interactions")
async def care_record(request: Request, patient_id: int) -> JSONResponse:
    import datetime as dt

    user = care_user(request)
    patient = next((p for p in PATIENTS if p["id"] == patient_id), None)
    if not user or patient is None:
        return JSONResponse({"error": "not allowed"}, status_code=403)
    body = await request.json()
    required = ("theme", "status", "direction", "date", "time")
    if any(not body.get(k) for k in required):
        return JSONResponse({"error": "missing fields"}, status_code=422)
    await asyncio.sleep(0.3)
    day = dt.date.fromisoformat(body["date"])
    log = {"patient": patient["name"], "theme": body["theme"], "status": body["status"],
           "direction": body["direction"], "start_date": day.strftime("%d %b %Y"), "start_time": body["time"],
           "iso_date": day.isoformat(), "notes": str(body.get("notes") or "")[:500], "recorded_by": user,
           "recorded": time.strftime("%d %b %Y %H:%M")}
    STATE.care_logs.setdefault(patient_id, []).insert(0, log)
    return JSONResponse(log)


@app.get("/care/api/patients/{patient_id}/interactions")
async def care_logs(patient_id: int) -> JSONResponse:
    return JSONResponse(STATE.care_logs.get(patient_id, []))


def main() -> None:
    import os

    import uvicorn

    uvicorn.run(app, host=os.environ.get("DEMO_HOST", "127.0.0.1"), port=8800, log_level="warning")


if __name__ == "__main__":
    main()
