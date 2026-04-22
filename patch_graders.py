import re

with open("environment/graders.py", "r", encoding="utf-8") as f:
    content = f.read()

new_explanation_func = """def get_underwriting_explanation(obs: "ApplicantProfile", risk: RiskLevel, dec: LoanDecision, rate: InterestRateTier) -> str:
    \"\"\"Generate a detailed explanation for the dynamic ground truth.\"\"\"
    parts = [
        f"Autonomous Bank Underwriting Pipeline for {obs.applicant_name}:",
        "",
        "1. Application & Documentation:",
        f"   - Applicant Details Received. Income: ₹{obs.annual_income:,.0f}, Requested Loan: ₹{obs.loan_amount_requested:,.0f}.",
        "   - Required documents: ID, Income Proof (Tax returns/Pay stubs), and Bank Statements.",
        "",
        "2. Loan Underwriting (5 C's of Credit):"
    ]
    
    # Character
    parts.append(f"   - Character: Credit score is {obs.credit_score}. Past defaults: {getattr(obs, 'previous_defaults', 0)}.")
    
    # Capacity
    dti = (obs.existing_debt / obs.annual_income * 100) if obs.annual_income > 0 else 0
    parts.append(f"   - Capacity: Debt-to-Income ratio is {dti:.1f}%.")
    
    # Capital & Collateral
    collateral_str = "Yes" if getattr(obs, 'has_collateral', False) else "No"
    parts.append(f"   - Capital & Collateral: Employment: {obs.employment_type.value}, Tenure: {getattr(obs, 'employment_years', 0)} yrs, Collateral: {collateral_str}.")
    
    # Conditions & Decision
    parts.append("")
    parts.append("3. Final Decision:")
    parts.append(f"   - Risk Assessment: {risk.value.upper()}")
    
    if dec == LoanDecision.APPROVE:
        parts.append(f"   - Result: {dec.value.upper()} - Strong financial indicators across the 5 C's.")
    elif dec == LoanDecision.REJECT:
        parts.append(f"   - Result: {dec.value.upper()} - Did not meet bank risk thresholds (high DTI or low credit).")
    else:
        parts.append(f"   - Result: {dec.value.upper()} - Pending submission and manual review of verified documents.")
        
    parts.append(f"   - Prescribed Interest Tier: {rate.value}")
    
    return "\\n".join(parts)"""

# Replace the old function
start_idx = content.find('def get_underwriting_explanation')
end_idx = content.find('def calculate_dynamic_ground_truth', start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_explanation_func + "\n\n\n" + content[end_idx:]
    with open("environment/graders.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched graders.py")
