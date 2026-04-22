import re

with open("static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# Replace executeStep back to a single call pipeline
execute_step_single = """// ─── Execute Pipeline ───────────────────────────────────────
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

    applicant.task_id = riskStage;

    // We do ONE comprehensive API call
    showToast(`🤖 AI Agent running full underwriting pipeline...`, 'info');

    const response = await fetch('/evaluate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(applicant)
    });

    if (!response.ok) {
        throw new Error(`Failed to evaluate application`);
    }

    const result = await response.json();
    
    // Display result
    displayResult(result);

    // Update lifecycle bar to show all completed
    activeStageId = riskStage;
    completedStages.add('lead_qualification_sales');
    completedStages.add('document_verification_hr');
    completedStages.add(riskStage);
    completedStages.add('customer_onboarding_pm');
    updateLifecycleBar();

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

start_idx = content.find("// ─── Execute Pipeline ───────────────────────────────────────")
if start_idx == -1:
    start_idx = content.find("async function executeStep() {")
end_idx = content.find("// ─── Display Result", start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + execute_step_single + content[end_idx:]

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched static/index.html for single call comprehensive pipeline")
