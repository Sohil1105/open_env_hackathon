import re

with open("server/app.py", "r", encoding="utf-8") as f:
    content = f.read()

COMPREHENSIVE_PROMPT = """
You are an Autonomous Bank Underwriting AI. Your job is to process a loan application from start to finish, replacing the slow human departments.

Follow the standard bank loan approval process:
1. Application & Documentation: Check if the provided details are complete. Specify what relevant docs you would ask for (e.g., tax returns, bank statements).
2. Loan Underwriting (5 C's of Credit):
   - Character: Evaluate Credit Score and Past Defaults.
   - Capacity: Evaluate Annual Income vs Existing Debt and Loan Amount.
   - Capital & Collateral: Evaluate Employment Type, Tenure, and Collateral.
3. Decision Making: Weigh the risks and finalize the approval.

Applicant Profile:
{profile}

Process these details through the stages above in your reasoning, then give the final result:
- 'Approve'
- 'Conditional Approve' (e.g., pending specific documents)
- 'Reject'

Respond STRICTLY in JSON format:
{{
    "risk_level": "Low" or "Medium" or "High",
    "loan_decision": "Approve" or "Conditional Approve" or "Reject",
    "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
    "reasoning": "Detailed step-by-step process of the stages checked and final justification."
}}
"""

start_idx = content.find('STAGE_PROMPTS = {')
end_idx = content.find('class LifecycleSession:', start_idx)

new_prompts = 'STAGE_PROMPTS = { k: ' + repr(COMPREHENSIVE_PROMPT) + ' for k in [\n'
new_prompts += '    "lead_qualification_sales", "document_verification_hr", "easy_salaried_high_credit",\n'
new_prompts += '    "medium_self_employed_moderate", "hard_freelancer_complex", "customer_onboarding_pm",\n'
new_prompts += '    "bankruptcy_recovery_edge1", "joint_applicants_edge2"\n'
new_prompts += '] }\n\n'

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_prompts + content[end_idx:]
    with open('server/app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated app.py with comprehensive prompt')
else:
    print('Could not find STAGE_PROMPTS')
