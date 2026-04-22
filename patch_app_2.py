import re

with open("server/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the fallback reasoning
content = re.sub(
    r'"reasoning": f"\[AI Error: \{str\(e\)}\] Using ground truth fallback\."',
    r'"reasoning": gt.explanation',
    content
)

# 2. Update `reset_environment`
reset_search = """        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    task_id = body.get("task_id", None)
                    custom_profile_data = body.get("custom_profile", None)
                    if custom_profile_data:
                        custom_profile = ApplicantProfile(**custom_profile_data)
            except Exception:
                pass"""

reset_replace = """        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    task_id = body.get("task_id", None)
                    
                    if task_id == TASK_ORDER[0]:
                        global_session.applicant_profile = {}
                        global_session.completed_stages = []
                        global_session.stage_scores = {}
                        
                    custom_profile_data = body.get("custom_profile", None)
                    if custom_profile_data:
                        custom_profile = ApplicantProfile(**custom_profile_data)
                    elif getattr(global_session, 'applicant_profile', None) and task_id != TASK_ORDER[0]:
                        try:
                            custom_profile = ApplicantProfile(**global_session.applicant_profile)
                        except Exception:
                            pass
            except Exception:
                pass"""

content = content.replace(reset_search, reset_replace)

# 3. Save profile to session in evaluate_applicant
# We will find `ground_truth = calculate_dynamic_ground_truth(profile)`
# and insert `global_session.applicant_profile = profile.model_dump()` before it.

gt_search = """    ground_truth = calculate_dynamic_ground_truth(profile)"""
gt_replace = """    global_session.applicant_profile = profile.model_dump()
    ground_truth = calculate_dynamic_ground_truth(profile)"""

content = content.replace(gt_search, gt_replace)

with open("server/app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched server/app.py")
