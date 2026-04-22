import re

with open("static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# Replace reasoning
content = content.replace(
    "typewriterEffect(document.getElementById('resReasoning'), decision.reasoning || 'No reasoning provided.');",
    """const reasoning = result.agent_decision?.reasoning
        || result.reasoning
        || result.explanation
        || 'No reasoning provided';
  
  typewriterEffect(document.getElementById('resReasoning'), reasoning);"""
)

# Add stage status if we want to display it somewhere, actually let's update resTaskName logic
content = content.replace(
    "document.getElementById('resTaskName').textContent = result.task_name || '—';",
    """document.getElementById('resTaskName').textContent = result.task_name || '—';
  
  const diffEl = document.getElementById('resDifficulty');
  const stageStatus = `Stage ${result.stage_number}/8 Complete`;
  const nextStageText = result.next_stage ? `Next: ${result.next_stage_name}` : 'Finished';
  // You can show this wherever fits best, maybe next to task name
  document.getElementById('resTaskName').textContent = `${result.task_name || '—'} (${stageStatus})`;
"""
)

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched index.html successfully.")
