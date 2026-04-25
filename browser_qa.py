"""
NEXUS BANK - Playwright Browser QA Suite
Comprehensive visual, functional, validation, reset, console, network, and accessibility checks.
"""

import io
import re
import sys
import time
import json
import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright, ConsoleMessage, Request, Response

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000/ui"
SERVER_BASE = "http://localhost:8000"
SCREENSHOT_DIR = Path(".")
REPORT_PATH = Path("browser_qa_report.txt")

# ─── Report Builder ──────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.pass_count = 0
        self.fail_count = 0
        self.warn_count = 0
        self.screenshots: list[str] = []
        self.console_errors: list[str] = []
        self.network_requests: list[dict] = []

    def _ts(self):
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def log(self, status: str, check_id: str, description: str, detail: str = ""):
        tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}.get(status, status)
        line = f"[{self._ts()}] {tag:<10} [{check_id}] {description}"
        if detail:
            line += f"\n           └─ {detail}"
        self.lines.append(line)
        print(line)
        if status == "PASS":
            self.pass_count += 1
        elif status == "FAIL":
            self.fail_count += 1
        elif status == "WARN":
            self.warn_count += 1

    def section(self, title: str):
        separator = "-" * 70
        header = f"\n{separator}\n  {title}\n{separator}"
        self.lines.append(header)
        print(header)

    def add_screenshot(self, filename: str):
        self.screenshots.append(filename)
        self.lines.append(f"           [SCREENSHOT] {filename}")

    def add_console_error(self, msg: str):
        self.console_errors.append(msg)
        self.lines.append(f"           [CONSOLE-ERR] {msg}")
        print(f"  [CONSOLE-ERR] {msg}")

    def add_network(self, req: dict):
        self.network_requests.append(req)

    def finalize(self) -> str:
        total = self.pass_count + self.fail_count + self.warn_count
        verdict = "PASS" if self.fail_count == 0 else "FAIL"
        summary = [
            "\n" + "=" * 70,
            "  FINAL SUMMARY",
            "=" * 70,
            f"  Total Checks : {total}",
            f"  [PASS]       : {self.pass_count}",
            f"  [FAIL]       : {self.fail_count}",
            f"  [WARN]       : {self.warn_count}",
            "",
            "  Screenshots:",
        ]
        for s in self.screenshots:
            summary.append(f"    - {s}")
        summary += [
            "",
            f"  Console Errors / Warnings: {len(self.console_errors)}",
        ]
        for e in self.console_errors:
            summary.append(f"    - {e}")
        summary += [
            "",
            f"  Network Requests ({len(self.network_requests)} total):",
        ]
        for r in self.network_requests:
            flag = " [SLOW]" if r.get("duration_ms", 0) > 20000 else ""
            flag += " [ERR]" if r.get("status", 0) >= 400 else ""
            summary.append(
                f"    {r['method']:<6} {r['status']:<5} {r['duration_ms']:>6}ms  {r['url']}{flag}"
            )
        summary += [
            "",
            f"  FINAL VERDICT: {verdict}",
            "=" * 70,
        ]
        block = "\n".join(summary)
        self.lines.append(block)
        return block


R = Report()


def ss(page, name: str) -> str:
    """Take a full-page screenshot and record it."""
    path = str(SCREENSHOT_DIR / name)
    page.screenshot(path=path, full_page=True)
    R.add_screenshot(name)
    return path


def wait_toast(page, timeout=4000) -> str:
    """Wait for a toast to become visible and return its text."""
    try:
        page.wait_for_selector(".toast.show", timeout=timeout)
        return page.locator(".toast").inner_text()
    except Exception:
        return ""


def clear_and_fill(page, selector: str, value: str):
    """Clear an input and fill with value."""
    loc = page.locator(selector)
    loc.click()
    loc.select_text()
    loc.fill(value)


# ─── Console + Network Collectors ────────────────────────────────────────────

console_log: list[str] = []
network_log: list[dict] = []
_pending: dict[str, float] = {}


def on_console(msg: ConsoleMessage):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    text = f"[{ts}] [{msg.type.upper()}] {msg.text}"
    if msg.type in ("error", "warning"):
        console_log.append(text)
        R.add_console_error(text)
    elif msg.type in ("log", "info", "debug"):
        # Only log non-trivial messages
        pass


def on_request(req: Request):
    _pending[req.url] = time.time()


def on_response(resp: Response):
    url = resp.url
    start = _pending.pop(url, time.time())
    dur = int((time.time() - start) * 1000)
    entry = {
        "method": resp.request.method,
        "url": url,
        "status": resp.status,
        "duration_ms": dur,
    }
    network_log.append(entry)
    R.add_network(entry)
    if resp.status >= 400:
        R.log("FAIL", "NET-ERR", f"HTTP {resp.status} for {resp.request.method} {url}")
    if dur > 20000:
        R.log("WARN", "NET-SLOW", f"Request took {dur}ms: {url}")


# ─── Main QA Runner ──────────────────────────────────────────────────────────

def run_qa():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            record_video_dir=None,
        )
        page = context.new_page()

        # Attach event listeners
        page.on("console", on_console)
        page.on("request", on_request)
        page.on("response", on_response)

        # ─── SUITE 1: Initial Load & Visual Checks ────────────────────────
        R.section("SUITE 1 — Initial Load & Visual Checks")

        page.goto(BASE_URL, wait_until="networkidle")
        time.sleep(1.5)  # Allow JS animations to settle
        ss(page, "screenshot_load.png")

        # 1.1 Page title
        title = page.title()
        if "NEXUS BANK" in title:
            R.log("PASS", "VIS-01", "Page title contains 'NEXUS BANK'", f"Title: '{title}'")
        else:
            R.log("FAIL", "VIS-01", "Page title missing 'NEXUS BANK'", f"Got: '{title}'")

        # 1.2 Wizard step indicators visible (1–4)
        for i in range(1, 5):
            loc = page.locator(f"#stepInd-{i}")
            if loc.is_visible():
                R.log("PASS", f"VIS-02-{i}", f"Wizard step indicator {i} is visible")
            else:
                R.log("FAIL", f"VIS-02-{i}", f"Wizard step indicator {i} is NOT visible")

        # 1.3 Form fields on step 1 render
        step1_fields = [
            ("#applicantName", "Applicant Name input"),
            ("#applicantAge", "Age input"),
            ("#employmentType", "Employment select"),
            ("#employmentYears", "Job History input"),
            ("#loanPurpose", "Loan Purpose input"),
        ]
        for sel, label in step1_fields:
            if page.locator(sel).is_visible():
                R.log("PASS", "VIS-03", f"Step 1 field visible: {label}")
            else:
                R.log("FAIL", "VIS-03", f"Step 1 field NOT visible: {label}", f"Selector: {sel}")

        # 1.4 Labels rendered for step 1 fields
        labels = page.locator("#wizardStep-1 .input-label").all()
        label_count = len(labels)
        if label_count >= 5:
            R.log("PASS", "VIS-04", f"Step 1 labels present: {label_count} found")
        else:
            R.log("FAIL", "VIS-04", f"Step 1 labels insufficient: {label_count} found (expected ≥5)")

        # 1.5 EXECUTE button hidden on step 1
        exec_btn = page.locator("#executeBtn")
        if not exec_btn.is_visible():
            R.log("PASS", "VIS-05", "EXECUTE (START UNDERWRITING) button hidden on Step 1")
        else:
            R.log("FAIL", "VIS-05", "EXECUTE button visible on Step 1 — should be hidden")

        # 1.6 NEXT button visible on step 1
        next_btn = page.locator("#nextBtn")
        if next_btn.is_visible():
            R.log("PASS", "VIS-06", "NEXT button visible on Step 1")
        else:
            R.log("FAIL", "VIS-06", "NEXT button NOT visible on Step 1")

        # 1.7 BACK button disabled on step 1
        prev_btn = page.locator("#prevBtn")
        if prev_btn.is_disabled():
            R.log("PASS", "VIS-07", "BACK button disabled on Step 1")
        else:
            R.log("FAIL", "VIS-07", "BACK button should be disabled on Step 1")

        # 1.8 Color theme — dark background
        bg_color = page.evaluate("() => window.getComputedStyle(document.body).backgroundColor")
        # #0a0a0f = rgb(10, 10, 15)
        if "10, 10, 15" in bg_color or "0a0a0f" in bg_color.lower():
            R.log("PASS", "VIS-08", "Dark background (#0a0a0f) confirmed", f"Got: {bg_color}")
        else:
            R.log("WARN", "VIS-08", "Background color may differ from expected #0a0a0f", f"Got: {bg_color}")

        # 1.9 Neon accent (primary color on status dot)
        primary_color = page.evaluate(
            "() => window.getComputedStyle(document.querySelector('.status-dot')).backgroundColor"
        )
        if "0, 255, 136" in primary_color:
            R.log("PASS", "VIS-09", "Neon primary accent #00ff88 confirmed on status dot", f"Got: {primary_color}")
        else:
            R.log("WARN", "VIS-09", "Primary neon color may differ", f"Got: {primary_color}")

        # 1.10 No broken images — check all img tags
        imgs = page.locator("img").all()
        broken = 0
        for img in imgs:
            natural_w = page.evaluate("el => el.naturalWidth", img.element_handle())
            if natural_w == 0:
                broken += 1
                R.log("FAIL", "VIS-10", "Broken image detected", f"src={img.get_attribute('src')}")
        if broken == 0:
            if len(imgs) == 0:
                R.log("PASS", "VIS-10", "No <img> tags found (icons are CSS/emoji, no broken images)")
            else:
                R.log("PASS", "VIS-10", f"All {len(imgs)} images load correctly")

        # 1.11 Particle canvas present
        canvas = page.locator("#particleCanvas")
        if canvas.count() > 0:
            R.log("PASS", "VIS-11", "Particle canvas (#particleCanvas) is present in DOM")
        else:
            R.log("FAIL", "VIS-11", "Particle canvas missing from DOM")

        # 1.12 Lifecycle bar has 8 stages
        stage_divs = page.locator(".lifecycle-stage").all()
        stage_count = len(stage_divs)
        if stage_count == 8:
            R.log("PASS", "VIS-12", f"Lifecycle bar has 8 stages as expected")
        else:
            R.log("FAIL", "VIS-12", f"Lifecycle bar stage count wrong: {stage_count} (expected 8)")

        # 1.13 System status text
        status_text = page.locator(".system-status span").inner_text()
        if "SYSTEM ONLINE" in status_text:
            R.log("PASS", "VIS-13", "System status reads 'SYSTEM ONLINE'")
        else:
            R.log("FAIL", "VIS-13", f"System status unexpected: '{status_text}'")

        # 1.14 NEXUS BANK glitch title
        title_el = page.locator(".glitch-title")
        if title_el.is_visible() and "NEXUS BANK" in title_el.inner_text():
            R.log("PASS", "VIS-14", "NEXUS BANK glitch title rendered")
        else:
            R.log("FAIL", "VIS-14", "NEXUS BANK glitch title missing or invisible")

        # 1.15 Global reset button
        if page.locator("#globalResetBtn").is_visible():
            R.log("PASS", "VIS-15", "Global RESET button visible in top bar")
        else:
            R.log("FAIL", "VIS-15", "Global RESET button NOT visible")

        # 1.16 Both panels visible (input + result)
        for panel_id, label in [("#inputPanel", "Terminal Input panel"), ("#resultPanel", "Decision Display panel")]:
            if page.locator(panel_id).is_visible():
                R.log("PASS", "VIS-16", f"{label} visible")
            else:
                R.log("FAIL", "VIS-16", f"{label} NOT visible")

        # ─── SUITE 2: Form Interaction / Full Happy Path ──────────────────
        R.section("SUITE 2 — Form Interaction: Happy Path (Full Underwriting Flow)")

        # Step 1 fill
        clear_and_fill(page, "#applicantName", "John Doe")
        clear_and_fill(page, "#applicantAge", "35")
        page.select_option("#employmentType", "salaried")
        clear_and_fill(page, "#employmentYears", "5")
        clear_and_fill(page, "#loanPurpose", "home improvement")
        R.log("PASS", "FORM-01", "Step 1 filled: name=John Doe, age=35, employment=salaried, purpose=home improvement")

        # Advance to Step 2
        page.click("#nextBtn")
        time.sleep(0.5)
        if page.locator("#wizardStep-2").evaluate("el => el.classList.contains('active')"):
            R.log("PASS", "FORM-02", "Advanced from Step 1 to Step 2")
        else:
            R.log("FAIL", "FORM-02", "Did not advance to Step 2 after clicking NEXT")

        # Step 2 fill: annual income, credit score, existing debt, prev defaults, public records, inquiries
        clear_and_fill(page, "#annualIncome", "85000")
        clear_and_fill(page, "#creditScore", "740")
        clear_and_fill(page, "#existingDebt", "30000")
        clear_and_fill(page, "#prevDefaults", "0")
        clear_and_fill(page, "#publicRecords", "0")
        clear_and_fill(page, "#inquiries", "0")
        R.log("PASS", "FORM-03", "Step 2 filled: income=85000, credit=740, debt=30000, defaults=0, records=0, inquiries=0")

        # Advance to Step 3
        page.click("#nextBtn")
        time.sleep(0.5)
        if page.locator("#wizardStep-3").evaluate("el => el.classList.contains('active')"):
            R.log("PASS", "FORM-04", "Advanced from Step 2 to Step 3")
        else:
            R.log("FAIL", "FORM-04", "Did not advance to Step 3")

        # Step 3 fill: loan amount, tenure
        clear_and_fill(page, "#loanAmount", "250000")
        clear_and_fill(page, "#loanTenure", "36")
        R.log("PASS", "FORM-05", "Step 3 filled: loan_amount=250000, tenure=36")

        # Advance to Step 4
        page.click("#nextBtn")
        time.sleep(0.5)
        if page.locator("#wizardStep-4").evaluate("el => el.classList.contains('active')"):
            R.log("PASS", "FORM-06", "Advanced from Step 3 to Step 4 (Document Upload)")
        else:
            R.log("FAIL", "FORM-06", "Did not advance to Step 4")

        # Verify EXECUTE button visible and NEXT hidden on step 4
        if exec_btn.is_visible():
            R.log("PASS", "FORM-07", "EXECUTE (START UNDERWRITING) button visible on Step 4")
        else:
            R.log("FAIL", "FORM-07", "EXECUTE button NOT visible on Step 4")

        if not next_btn.is_visible():
            R.log("PASS", "FORM-08", "NEXT button correctly hidden on Step 4")
        else:
            R.log("FAIL", "FORM-08", "NEXT button should be hidden on Step 4")

        # EXECUTE — click and wait for result
        R.log("INFO", "FORM-09", "Clicking START UNDERWRITING - this may take 10-30s for LLM response...")
        page.click("#executeBtn")

        # Wait up to 60s for result content to appear
        try:
            page.wait_for_selector("#resultContent", state="visible", timeout=90000)
            time.sleep(2)  # Allow typewriter effect to start
            result_visible = True
        except Exception as e:
            result_visible = False
            R.log("FAIL", "FORM-09", "Result content never appeared after executing", str(e))

        ss(page, "screenshot_result.png")

        if result_visible:
            R.log("PASS", "FORM-09", "Result panel became visible after EXECUTE")

            # Verify result fields populated
            for field_id, label in [
                ("resRiskLevel", "risk_level"),
                ("resDecision", "loan_decision"),
                ("resInterestRate", "interest_rate_tier"),
                ("resReasoning", "reasoning"),
            ]:
                val = page.locator(f"#{field_id}").inner_text().strip()
                if val and val != "—" and val != "Awaiting decision...":
                    R.log("PASS", "FORM-10", f"Result field '{label}' populated", f"Value: '{val[:80]}{'…' if len(val) > 80 else ''}'")
                else:
                    R.log("FAIL", "FORM-10", f"Result field '{label}' is empty or default", f"Got: '{val}'")

            # Verify Approve = green color #00ff88
            decision_text = page.locator("#resDecision").inner_text().strip()
            decision_color = page.evaluate(
                "() => window.getComputedStyle(document.getElementById('resDecision')).color"
            )
            if decision_text == "Approve":
                if "0, 255, 136" in decision_color:
                    R.log("PASS", "FORM-11", "Approve decision rendered in green #00ff88", f"Color: {decision_color}")
                else:
                    R.log("FAIL", "FORM-11", "Approve decision color is NOT #00ff88", f"Got: {decision_color}")
            elif decision_text in ("Reject", "Conditional Approve"):
                R.log("INFO", "FORM-11", f"Decision is '{decision_text}' (not Approve) — color check adjusted")
                if "Conditional Approve" in decision_text and "255, 204, 0" in decision_color:
                    R.log("PASS", "FORM-11b", "Conditional Approve rendered in gold #ffcc00")
                elif "Reject" in decision_text and "255, 51, 102" in decision_color:
                    R.log("PASS", "FORM-11b", "Reject rendered in red/danger #ff3366")
                else:
                    R.log("WARN", "FORM-11b", f"Decision color may be incorrect", f"Decision: '{decision_text}', Color: {decision_color}")
            else:
                R.log("WARN", "FORM-11", f"Unexpected decision text: '{decision_text}'")

            # Verify score bar renders and width > 0
            bar_width = page.evaluate(
                "() => document.getElementById('resScoreBar').style.width"
            )
            bar_width_str = bar_width.replace("%", "").strip()
            try:
                bar_pct = float(bar_width_str)
                if bar_pct > 0:
                    R.log("PASS", "FORM-12", f"Score bar has width > 0%", f"Width: {bar_pct:.1f}%")
                else:
                    R.log("FAIL", "FORM-12", "Score bar width is 0% — not rendered", f"Width: {bar_width}")
            except ValueError:
                R.log("FAIL", "FORM-12", "Score bar width could not be parsed", f"Raw: '{bar_width}'")

            # Verify lifecycle bar 8 stages after execution
            stage_divs_after = page.locator(".lifecycle-stage").all()
            if len(stage_divs_after) == 8:
                R.log("PASS", "FORM-13", "Lifecycle bar shows 8 stages after execution")
            else:
                R.log("FAIL", "FORM-13", f"Lifecycle bar stage count after execute: {len(stage_divs_after)}")

            # Check at least some stages are marked completed
            completed_stages = page.locator(".stage-number.completed").count()
            if completed_stages > 0:
                R.log("PASS", "FORM-14", f"Lifecycle bar shows {completed_stages} completed stages")
            else:
                R.log("WARN", "FORM-14", "No lifecycle stages marked as completed after execution")

            # Ground truth box visible
            gt_box = page.locator("#groundTruthBox")
            if gt_box.is_visible():
                R.log("PASS", "FORM-15", "Ground Truth box is visible")
            else:
                R.log("WARN", "FORM-15", "Ground Truth box is not visible after execution")

        # ─── SUITE 3: Validation / Negative Tests ─────────────────────────
        R.section("SUITE 3 — Validation & Negative Tests")

        # Reset first to get back to step 1
        page.click("#globalResetBtn")
        time.sleep(1)

        # Validation Test 1: Submit empty name on Step 1
        clear_and_fill(page, "#applicantName", "")
        page.click("#nextBtn")
        time.sleep(0.5)
        toast_text = wait_toast(page, timeout=3000)
        ss(page, "screenshot_validation_1.png")
        if toast_text and ("name" in toast_text.lower() or "applicant" in toast_text.lower()):
            R.log("PASS", "VAL-01", "Empty name → toast error appears", f"Toast: '{toast_text}'")
        elif toast_text:
            R.log("WARN", "VAL-01", "Toast shown for empty name but message may be wrong", f"Toast: '{toast_text}'")
        else:
            still_step1 = page.locator("#wizardStep-1").evaluate("el => el.classList.contains('active')")
            if still_step1:
                R.log("WARN", "VAL-01", "Empty name blocked navigation but toast not detected")
            else:
                R.log("FAIL", "VAL-01", "Empty name did NOT block navigation — no validation")

        # Validation Test 2: Fill name, advance to Step 2, try FICO=999
        clear_and_fill(page, "#applicantName", "Test User")
        page.click("#nextBtn")
        time.sleep(0.5)

        clear_and_fill(page, "#annualIncome", "70000")
        clear_and_fill(page, "#creditScore", "999")  # Invalid FICO
        clear_and_fill(page, "#existingDebt", "0")
        page.click("#nextBtn")
        time.sleep(0.5)
        toast_text2 = wait_toast(page, timeout=3000)
        ss(page, "screenshot_validation_2.png")
        if toast_text2 and ("fico" in toast_text2.lower() or "300" in toast_text2 or "850" in toast_text2 or "credit" in toast_text2.lower()):
            R.log("PASS", "VAL-02", "FICO=999 → toast error appears", f"Toast: '{toast_text2}'")
        elif toast_text2:
            R.log("WARN", "VAL-02", "Toast shown for FICO=999 but message unexpected", f"Toast: '{toast_text2}'")
        else:
            still_step2 = page.locator("#wizardStep-2").evaluate("el => el.classList.contains('active')")
            if still_step2:
                R.log("WARN", "VAL-02", "FICO=999 blocked progression but no toast detected")
            else:
                R.log("FAIL", "VAL-02", "FICO=999 was NOT blocked — invalid FICO accepted")

        # Fix FICO, advance to step 3, test loan_amount=0
        clear_and_fill(page, "#creditScore", "700")
        page.click("#nextBtn")
        time.sleep(0.5)

        clear_and_fill(page, "#loanAmount", "0")
        page.click("#nextBtn")
        time.sleep(0.5)
        toast_text3 = wait_toast(page, timeout=3000)
        ss(page, "screenshot_validation_3.png")
        if toast_text3 and ("loan" in toast_text3.lower() or "amount" in toast_text3.lower() or "greater" in toast_text3.lower()):
            R.log("PASS", "VAL-03", "loan_amount=0 → toast error appears", f"Toast: '{toast_text3}'")
        elif toast_text3:
            R.log("WARN", "VAL-03", "Toast shown for loan=0 but message unexpected", f"Toast: '{toast_text3}'")
        else:
            still_step3 = page.locator("#wizardStep-3").evaluate("el => el.classList.contains('active')")
            if still_step3:
                R.log("WARN", "VAL-03", "loan=0 blocked progression but no toast detected")
            else:
                R.log("FAIL", "VAL-03", "loan_amount=0 was NOT blocked")

        # Validation Test 4: Negative debt — go back to step 2 and test
        # First reset and navigate fresh
        page.click("#globalResetBtn")
        time.sleep(0.8)

        clear_and_fill(page, "#applicantName", "Test User")
        page.click("#nextBtn")
        time.sleep(0.5)

        clear_and_fill(page, "#annualIncome", "70000")
        clear_and_fill(page, "#creditScore", "700")
        clear_and_fill(page, "#existingDebt", "-5000")  # Negative debt
        page.click("#nextBtn")
        time.sleep(0.5)
        toast_text4 = wait_toast(page, timeout=3000)
        ss(page, "screenshot_validation_4.png")
        if toast_text4 and ("debt" in toast_text4.lower() or "negative" in toast_text4.lower()):
            R.log("PASS", "VAL-04", "Negative debt → toast error appears", f"Toast: '{toast_text4}'")
        elif toast_text4:
            R.log("WARN", "VAL-04", "Toast shown for negative debt but message unexpected", f"Toast: '{toast_text4}'")
        else:
            still_step2 = page.locator("#wizardStep-2").evaluate("el => el.classList.contains('active')")
            if still_step2:
                R.log("WARN", "VAL-04", "Negative debt blocked navigation but no toast detected")
            else:
                R.log("FAIL", "VAL-04", "Negative debt was NOT blocked — accepted as valid")

        # ─── SUITE 4: Reset Checks ────────────────────────────────────────
        R.section("SUITE 4 — Global Reset Check")

        # Navigate back to step 1 and do a full flow then reset
        page.click("#globalResetBtn")
        time.sleep(0.8)

        # Fill step 1 data
        clear_and_fill(page, "#applicantName", "Reset Test User")
        clear_and_fill(page, "#loanPurpose", "car")
        page.click("#nextBtn")
        time.sleep(0.4)

        # Quick fill step 2
        clear_and_fill(page, "#annualIncome", "50000")
        clear_and_fill(page, "#creditScore", "650")
        clear_and_fill(page, "#existingDebt", "1000")
        page.click("#nextBtn")
        time.sleep(0.4)

        # Quick fill step 3
        clear_and_fill(page, "#loanAmount", "100000")
        page.click("#nextBtn")
        time.sleep(0.4)

        # Now click global reset
        page.click("#globalResetBtn")
        time.sleep(1)

        # Verify form cleared
        name_val = page.locator("#applicantName").input_value()
        income_val = page.evaluate("() => document.getElementById('annualIncome')?.value || ''")
        credit_val = page.evaluate("() => document.getElementById('creditScore')?.value || ''")

        if name_val == "":
            R.log("PASS", "RST-01", "applicantName cleared after global reset")
        else:
            R.log("FAIL", "RST-01", f"applicantName NOT cleared after reset", f"Got: '{name_val}'")

        if income_val == "":
            R.log("PASS", "RST-02", "annualIncome cleared after global reset")
        else:
            R.log("FAIL", "RST-02", f"annualIncome NOT cleared after reset", f"Got: '{income_val}'")

        # Verify back to Step 1
        on_step1 = page.locator("#wizardStep-1").evaluate("el => el.classList.contains('active')")
        if on_step1:
            R.log("PASS", "RST-03", "Wizard returned to Step 1 after global reset")
        else:
            R.log("FAIL", "RST-03", "Wizard NOT on Step 1 after global reset")

        # Verify result panel cleared (waiting state shown)
        waiting_state = page.locator("#waitingState")
        result_content = page.locator("#resultContent")
        if waiting_state.is_visible():
            R.log("PASS", "RST-04", "Waiting state shown after reset (result panel cleared)")
        else:
            R.log("FAIL", "RST-04", "Waiting state NOT visible after reset")

        if not result_content.is_visible():
            R.log("PASS", "RST-05", "Result content hidden after reset")
        else:
            R.log("FAIL", "RST-05", "Result content still visible after reset (should be hidden)")

        # Execute button hidden after reset
        if not exec_btn.is_visible():
            R.log("PASS", "RST-06", "EXECUTE button hidden after reset (on step 1)")
        else:
            R.log("FAIL", "RST-06", "EXECUTE button still visible after reset on step 1")

        ss(page, "screenshot_reset.png")

        # ─── SUITE 5: Accessibility Checks ───────────────────────────────
        R.section("SUITE 5 — Accessibility Checks")

        # Check all inputs have label or placeholder
        all_inputs = page.locator("input[type='text'], input[type='number'], input:not([type])").all()
        inputs_ok = 0
        inputs_fail = 0
        for inp in all_inputs:
            inp_id = inp.get_attribute("id") or ""
            placeholder = inp.get_attribute("placeholder") or ""
            # Check for associated label
            has_label = False
            if inp_id:
                label_count = page.locator(f"label[for='{inp_id}']").count()
                # Also check parent .input-row for .input-label
                has_label = label_count > 0 or page.evaluate(
                    f"() => document.getElementById('{inp_id}')?.closest('.input-row')?.querySelector('.input-label') !== null"
                )
            if has_label or placeholder:
                inputs_ok += 1
            else:
                inputs_fail += 1
                R.log("WARN", "ACC-01", f"Input #{inp_id} has no label or placeholder")

        if inputs_fail == 0:
            R.log("PASS", "ACC-01", f"All {inputs_ok} inputs have labels or placeholders")
        else:
            R.log("FAIL", "ACC-01", f"{inputs_fail} inputs missing labels/placeholders")

        # Check buttons have readable text
        buttons = page.locator("button").all()
        btn_ok = 0
        btn_fail = 0
        for btn_el in buttons:
            txt = (btn_el.inner_text() or "").strip()
            if txt:
                btn_ok += 1
            else:
                btn_fail += 1
                R.log("WARN", "ACC-02", "Button with no readable text found", f"HTML: {btn_el.inner_html()[:60]}")

        if btn_fail == 0:
            R.log("PASS", "ACC-02", f"All {btn_ok} buttons have readable text")
        else:
            R.log("WARN", "ACC-02", f"{btn_fail} buttons have no readable text")

        # Check no input has type="password" accidentally
        pw_inputs = page.locator("input[type='password']").count()
        if pw_inputs == 0:
            R.log("PASS", "ACC-03", "No accidental password-type inputs found")
        else:
            R.log("FAIL", "ACC-03", f"Found {pw_inputs} password-type input(s) unexpectedly")

        # Tab order — test tabbing through step 1 fields
        page.keyboard.press("Tab")
        focused_id_1 = page.evaluate("() => document.activeElement?.id || ''")
        page.keyboard.press("Tab")
        focused_id_2 = page.evaluate("() => document.activeElement?.id || ''")
        if focused_id_1 or focused_id_2:
            R.log("PASS", "ACC-04", f"Tab key navigates between focusable elements", f"First: #{focused_id_1}, Second: #{focused_id_2}")
        else:
            R.log("WARN", "ACC-04", "Could not verify tab order — no focused elements detected")

        # Check BACK button is not just an icon without aria-label
        prev_text = page.locator("#prevBtn").inner_text().strip()
        next_text = page.locator("#nextBtn").inner_text().strip()
        exec_text = page.locator("#executeBtn").inner_text().strip()
        for btn_name, txt in [("BACK", prev_text), ("NEXT", next_text), ("EXECUTE", exec_text)]:
            if txt:
                R.log("PASS", "ACC-05", f"Navigation button '{btn_name}' has readable text: '{txt}'")
            else:
                R.log("FAIL", "ACC-05", f"Navigation button '{btn_name}' has no readable text")

        # ─── SUITE 6: Console Error Summary ──────────────────────────────
        R.section("SUITE 6 — Console Error / Warning Summary")

        if len(console_log) == 0:
            R.log("PASS", "CON-01", "No JS console errors or warnings detected during session")
        else:
            R.log("FAIL", "CON-01", f"{len(console_log)} console errors/warnings collected during session")
            for msg in console_log:
                R.log("INFO", "CON-01", "Console message", msg)

        # ─── SUITE 7: Network Request Summary ────────────────────────────
        R.section("SUITE 7 — Network Request Summary")

        api_requests = [r for r in network_log if SERVER_BASE in r["url"]]
        R.log("INFO", "NET-01", f"Total network requests captured: {len(network_log)}")
        R.log("INFO", "NET-01", f"API (server) requests: {len(api_requests)}")

        errors_4xx5xx = [r for r in network_log if r.get("status", 0) >= 400]
        slow_reqs = [r for r in network_log if r.get("duration_ms", 0) > 20000]

        if len(errors_4xx5xx) == 0:
            R.log("PASS", "NET-02", "No 4xx/5xx HTTP errors in any network request")
        else:
            R.log("FAIL", "NET-02", f"{len(errors_4xx5xx)} HTTP error(s) detected")
            for r in errors_4xx5xx:
                R.log("FAIL", "NET-02", f"HTTP {r['status']}: {r['method']} {r['url']}")

        if len(slow_reqs) == 0:
            R.log("PASS", "NET-03", "No requests exceeded 20 000ms timeout")
        else:
            R.log("WARN", "NET-03", f"{len(slow_reqs)} slow request(s) exceeded 20 000ms")
            for r in slow_reqs:
                R.log("WARN", "NET-03", f"SLOW {r['duration_ms']}ms: {r['method']} {r['url']}")

        # List all API requests
        for r in api_requests:
            flag = ""
            if r.get("status", 0) >= 400:
                flag = " ← ERROR"
            if r.get("duration_ms", 0) > 20000:
                flag += " ← SLOW"
            R.log("INFO", "NET-04", f"{r['method']} {r['url']} → {r['status']} ({r['duration_ms']}ms){flag}")

        # ─── SUITE 8: Additional UI / Edge Checks ────────────────────────
        R.section("SUITE 8 — Additional UI & Edge Checks")

        # Viewport / responsive check
        page.set_viewport_size({"width": 375, "height": 812})
        time.sleep(0.5)
        body_overflow = page.evaluate("() => document.body.scrollWidth > window.innerWidth")
        if body_overflow:
            R.log("WARN", "RESP-01", "Horizontal scroll detected on mobile viewport (375px) — possible layout overflow")
        else:
            R.log("PASS", "RESP-01", "No horizontal overflow on mobile viewport (375px)")

        # Restore viewport
        page.set_viewport_size({"width": 1400, "height": 900})
        time.sleep(0.3)

        # File upload zone clickable and shows toast
        page.click("#globalResetBtn")
        time.sleep(0.5)
        # Navigate to step 4
        clear_and_fill(page, "#applicantName", "Upload Test")
        page.click("#nextBtn"); time.sleep(0.3)
        clear_and_fill(page, "#annualIncome", "60000")
        clear_and_fill(page, "#creditScore", "700")
        clear_and_fill(page, "#existingDebt", "0")
        page.click("#nextBtn"); time.sleep(0.3)
        clear_and_fill(page, "#loanAmount", "50000")
        page.click("#nextBtn"); time.sleep(0.5)

        upload_zone = page.locator(".file-upload-zone")
        if upload_zone.is_visible():
            R.log("PASS", "UI-01", "File upload zone is visible on Step 4")
            upload_zone.click()
            time.sleep(0.5)
            toast_upload = wait_toast(page, timeout=3000)
            file_items = page.locator(".file-item").count()
            if file_items > 0 or "uploaded" in toast_upload.lower():
                R.log("PASS", "UI-02", "File upload simulation works — file item appeared", f"Toast: '{toast_upload}'")
            else:
                R.log("WARN", "UI-02", "File upload click did not produce visible file item", f"Toast: '{toast_upload}'")
        else:
            R.log("FAIL", "UI-01", "File upload zone NOT visible on Step 4")

        # Step indicators update correctly when navigating
        page.click("#globalResetBtn"); time.sleep(0.5)
        step1_active = page.locator("#stepInd-1").evaluate("el => el.classList.contains('active')")
        if step1_active:
            R.log("PASS", "UI-03", "Step 1 indicator is active after reset")
        else:
            R.log("FAIL", "UI-03", "Step 1 indicator NOT active after reset")

        # Navigate to step 2 and verify indicator updates
        clear_and_fill(page, "#applicantName", "Nav Test")
        page.click("#nextBtn"); time.sleep(0.4)
        step1_completed = page.locator("#stepInd-1").evaluate("el => el.classList.contains('completed')")
        step2_active = page.locator("#stepInd-2").evaluate("el => el.classList.contains('active')")
        if step1_completed:
            R.log("PASS", "UI-04", "Step 1 indicator marked 'completed' after advancing to Step 2")
        else:
            R.log("WARN", "UI-04", "Step 1 indicator not marked 'completed' after advancing")
        if step2_active:
            R.log("PASS", "UI-05", "Step 2 indicator marked 'active' after advancing")
        else:
            R.log("FAIL", "UI-05", "Step 2 indicator NOT marked 'active' after advancing")

        # Lifecycle bar connectors present (7 connectors for 8 stages)
        connectors = page.locator(".stage-connector").count()
        if connectors == 7:
            R.log("PASS", "UI-06", "Lifecycle bar has 7 connectors (correct for 8 stages)")
        else:
            R.log("WARN", "UI-06", f"Lifecycle bar connector count: {connectors} (expected 7)")

        # Score bar initial width is 0%
        page.click("#globalResetBtn"); time.sleep(0.5)
        score_bar_w = page.evaluate("() => document.getElementById('resScoreBar').style.width")
        if score_bar_w == "0%":
            R.log("PASS", "UI-07", "Score bar correctly shows 0% after reset")
        else:
            R.log("WARN", "UI-07", f"Score bar after reset: '{score_bar_w}' (expected '0%')")

        # ENV label in top bar
        env_label = page.locator(".env-label").inner_text()
        if "loan-lifecycle" in env_label.lower() or "env" in env_label.lower():
            R.log("PASS", "UI-08", f"ENV label present in top bar: '{env_label}'")
        else:
            R.log("WARN", "UI-08", f"ENV label unexpected: '{env_label}'")

        # Subtitle text
        subtitle = page.locator(".subtitle").inner_text()
        if subtitle:
            R.log("PASS", "UI-09", f"Subtitle text present: '{subtitle}'")
        else:
            R.log("FAIL", "UI-09", "Subtitle text missing")

        # ─── DONE ─────────────────────────────────────────────────────────
        browser.close()

    # Finalize report
    summary = R.finalize()
    print(summary)

    # Write full report to file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"NEXUS BANK — Browser QA Report\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target: {BASE_URL}\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(R.lines))

    print(f"\n[DONE] Full report written to: {REPORT_PATH}")
    return R.fail_count == 0


if __name__ == "__main__":
    ok = run_qa()
    sys.exit(0 if ok else 1)
