import re

with open("server/app.py", "r", encoding="utf-8") as f:
    content = f.read()

COMPREHENSIVE_PROMPT = """
You are an Advanced Autonomous Bank Underwriting Agent. Your mission is to replace the manual, multi-department loan approval process with a high-speed AI pipeline.

Perform the full evaluation by processing the applicant through these 5 professional banking stages:

STAGE 1: DOCUMENTATION & IDENTITY VERIFICATION
- Review the applicant's profile for completeness.
- List specific documents required for this profile (e.g., if self-employed, ask for 2 years of tax returns; if salaried, ask for recent pay stubs).

STAGE 2: CREDIT CHARACTER ASSESSMENT
- Evaluate the Credit Score ({credit_score}) and Past Defaults ({past_defaults}). 
- Analyze their reliability as a borrower.

STAGE 3: CAPACITY & CAPITAL ANALYSIS
- Calculate the Debt-to-Income (DTI) ratio (Existing Debt: {existing_debt} / Annual Income: {annual_income}).
- Assess if their income provides enough "Capacity" to repay the requested Loan Amount ({loan_amount}).
- Consider their "Capital" (employment stability and years of experience).

STAGE 4: COLLATERAL & CONDITIONS
- Evaluate if "Collateral" ({has_collateral}) is provided to secure the loan.
- Check "Conditions": Is the loan tenure ({loan_tenure} months) and interest rate tier appropriate for current market conditions?

STAGE 5: FINAL UNDERWRITING DECISION
- Synthesize all findings into a final Risk Level and Loan Decision.

Applicant Profile:
{profile}

Respond STRICTLY in JSON format with this structure:
{{
    "risk_level": "Low" | "Medium" | "High",
    "loan_decision": "Approve" | "Conditional Approve" | "Reject",
    "interest_rate_tier": "7-9%" | "10-13%" | "14%+",
    "requested_documents": ["list", "of", "required", "docs"],
    "reasoning": "A comprehensive report covering all 5 stages in detail."
}}
"""

# Update STAGE_PROMPTS
start_idx = content.find('STAGE_PROMPTS = {')
end_idx = content.find('class LifecycleSession:', start_idx)

new_prompts = 'STAGE_PROMPTS = { k: ' + repr(COMPREHENSIVE_PROMPT) + ' for k in [\n'
new_prompts += '    "lead_qualification_sales", "document_verification_hr", "easy_salaried_high_credit",\n'
new_prompts += '    "medium_self_employed_moderate", "hard_freelancer_complex", "customer_onboarding_pm",\n'
new_prompts += '    "bankruptcy_recovery_edge1", "joint_applicants_edge2"\n'
new_prompts += '] }\n\n'

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_prompts + content[end_idx:]

# Update the profile formatting in evaluate_applicant to include the extra variables for the prompt
profile_formatting_search = """    full_prompt = prompt_template.format(profile=profile_text)"""
profile_formatting_replace = """    full_prompt = prompt_template.format(
        profile=profile_text,
        credit_score=applicant.credit_score,
        past_defaults=getattr(applicant, 'previous_defaults', 0),
        existing_debt=applicant.existing_debt,
        annual_income=applicant.annual_income,
        loan_amount=applicant.loan_amount,
        has_collateral="Provided" if getattr(applicant, 'has_collateral', False) else "None",
        loan_tenure=applicant.loan_tenure
    )"""

content = content.replace(profile_formatting_search, profile_formatting_replace)

with open('server/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated app.py with ultra-comprehensive agentic prompt')
