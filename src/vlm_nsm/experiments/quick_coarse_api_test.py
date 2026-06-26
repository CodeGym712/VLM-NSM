"""Quick API test for coarse DSL pipeline."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser
from vlm_nsm.neuro_symbolic.verifier_bank import CoarseOptionVerifier

API_KEY = os.environ.get("QWEN_API_KEY", "")
if not API_KEY:
    sys.exit("Error: QWEN_API_KEY environment variable is not set. "
             "Obtain a key from Alibaba Cloud Model Studio and try again.")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

parser = OnlineQwenGoalParser(
    model='qwen-plus', api_key=API_KEY, api_base=API_BASE,
    temperature=0.0, max_tokens=512, retries=1)

v = CoarseOptionVerifier(valid_objects=['cube1', 'cube2', 'cube3', 'table'])

tests = [
    ('Stack cube1 on cube2', ['cube1','cube2','table'], 2),
    ('Pick up cube1 and put it on cube2', ['cube1','cube2','table'], 2),
    ('Stack cube1 on cube2 then cube2 on cube3', ['cube1','cube2','cube3','table'], 3),
    ('Unstack cube1 and put it on the table', ['cube1','cube2','table'], 2),
    ('Make a tower: cube3 base, cube2 middle, cube1 top', ['cube1','cube2','cube3','table'], 3),
    ('Move cube2 from cube1 to the table', ['cube1','cube2','table'], 2),
]

results = []
for instruction, objects, n_blocks in tests:
    r = parser.parse(instruction, objects=objects, task_type='coarse_stacking', n_blocks=n_blocks)
    steps = r.goal_json.get('steps', []) if r.goal_json else []
    report = v.check(coarse_steps=steps)
    forbidden_ops = {'reach', 'grasp', 'lift', 'reach_place', 'release_place', 'retreat'}
    has_forbidden = any(isinstance(s, dict) and s.get('op', '') in forbidden_ops for s in steps)

    entry = {
        'instruction': instruction, 'n_blocks': n_blocks,
        'parse_ok': r.success, 'n_steps': len(steps),
        'verify_passed': report.passed, 'has_forbidden': has_forbidden,
        'steps': steps, 'failures': [(f[0], f[1]) for f in report.failures],
    }
    results.append(entry)

    print(f"[{instruction}] parse={r.success} steps={len(steps)} verify={report.passed} forbid={has_forbidden}")
    for s in steps:
        op, obj = s.get('op','?'), s.get('object','')
        tgt = s.get('target',''); pred = s.get('predicate',[])
        if op in ('pick','place'): print(f"  {op}({obj}{', ' + tgt if tgt else ''})")
        elif op in ('ensure','achieve'): print(f"  {op}({pred})")
        else: print(f"  {op}({obj})")
    if not report.passed:
        for f in report.failures[:3]: print(f"  FAIL: {f[0]}: {f[1]}")
    print()

# Summary
n_ok = sum(1 for r in results if r['parse_ok'] and r['verify_passed'] and not r['has_forbidden'])
print(f"Summary: {n_ok}/{len(tests)} fully valid coarse programs")

# Save
out_dir = 'generated/results/coarse_dsl_benchmark'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/api_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_dir}/api_test_results.json")
