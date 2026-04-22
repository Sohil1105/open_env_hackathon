import re

with open("static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Hide the stage select dropdown
content = content.replace(
    """<div class="input-row">
        <label class="input-label" for="stageSelect">
          <span class="prompt-char">›</span> Stage Select
        </label>
        <select id="stageSelect" class="select-field" onchange="onStageSelect()">
          <option value="">— Select Stage —</option>
        </select>
      </div>""",
    """<div class="input-row" style="display:none;">
        <select id="stageSelect" class="select-field"></select>
      </div>
      <div class="input-row">
        <label class="input-label" style="color:var(--accent);">
          <span class="prompt-char">›</span> Pipeline Mode
        </label>
        <div class="input-field" style="display:flex; align-items:center; background:rgba(0,255,136,0.1); color:var(--accent); border:1px solid var(--accent);">
          🚀 AUTONOMOUS AI UNDERWRITER ACTIVE
        </div>
      </div>"""
)

# 2. Replace executeStep function to be a full pipeline runner
# I need to find the `async function executeStep() {` and replace it
# Wait, it's safer to just replace the whole function

execute_step_replacement = """// ─── Execute Pipeline ───────────────────────────────────────
async function executeStep() {
  const btn = document.getElementById('executeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>AI ANALYZING PIPELINE...';

  try {
    // Collect applicant details
    const applicant = {
      applicant_name: document.getElementById('applicantName').value,
      age: parseInt(document.getElementById('applicantAge')?.value || 30),
      annual_income: parseFloat(document.getElementById('annualIncome').value) || 0,
      credit_score: parseInt(document.getElementById('creditScore').value) || 0,
      existing_debt: parseFloat(document.getElementById('existingDebt').value) || 0,
      loan_amount: parseFloat(document.getElementById('loanAmount').value) || 0,
      employment_type: document.getElementById('employmentType').value,
      employment_years: parseFloat(document.getElementById('employmentYears')?.value || 2),
      previous_defaults: parseInt(document.getElementById('prevDefaults')?.value || 0),
      loan_tenure: parseInt(document.getElementById('loanTenure').value) || 36
    };

    if (!applicant.applicant_name || !applicant.annual_income || !applicant.loan_amount) {
        showToast('Please fill in Name, Income, and Loan Amount', 'error');
        btn.disabled = false;
        btn.innerHTML = '▶ EXECUTE AUTOPILOT';
        return;
    }

    // Determine risk stage based on employment type
    let riskStage = 'easy_salaried_high_credit';
    if (applicant.employment_type === 'self_employed' || applicant.employment_type === 'contract') {
        riskStage = 'medium_self_employed_moderate';
    } else if (applicant.employment_type === 'freelancer' || applicant.employment_type === 'unemployed') {
        riskStage = 'hard_freelancer_complex';
    }

    const pipeline = [
      { id: 'lead_qualification_sales', name: 'Lead Qualification' },
      { id: 'document_verification_hr', name: 'Document Verification' },
      { id: riskStage, name: 'Risk Assessment' },
      { id: 'customer_onboarding_pm', name: 'Customer Onboarding' }
    ];

    completedStages.clear();

    for (let i = 0; i < pipeline.length; i++) {
        const stage = pipeline[i];
        activeStageId = stage.id;
        updateLifecycleBar();
        
        showToast(`🤖 AI Agent running: ${stage.name}...`, 'info');
        btn.innerHTML = `<span class="spinner"></span>RUNNING ${stage.name.toUpperCase()}...`;

        applicant.task_id = stage.id;
        
        // Wait 1 second for visual effect
        await new Promise(r => setTimeout(r, 1000));

        const response = await fetch('/evaluate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(applicant)
        });

        if (!response.ok) {
            throw new Error(`Failed at ${stage.name}`);
        }

        const result = await response.json();
        
        // Display result for this stage
        displayResult(result);
        completedStages.add(stage.id);
        updateLifecycleBar();

        const decision = result.agent_decision?.loan_decision || result.ground_truth?.loan_decision;
        
        if (decision === 'Reject' || decision === 'Disqualify') {
            showToast(`Application rejected at ${stage.name}`, 'error');
            break; // Stop pipeline
        }
        
        // Wait 2 seconds before next stage so user can read
        if (i < pipeline.length - 1) {
            await new Promise(r => setTimeout(r, 2500));
        }
    }

    showToast('✨ Autonomous Pipeline Complete!', 'success');

  } catch (e) {
    console.error('Execute failed:', e);
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ EXECUTE AUTOPILOT';
  }
}
"""

# Find executeStep in content
start_idx = content.find("async function executeStep() {")
end_idx = content.find("// ─── Display Result", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + execute_step_replacement + content[end_idx:]

# Change button text
content = content.replace("▶ EXECUTE</button>", "▶ EXECUTE AUTOPILOT</button>")

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched static/index.html for pipeline automation")
