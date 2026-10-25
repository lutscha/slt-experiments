"""Print which camera-ready runs have finished. Usage: python camera_ready/status.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run  # noqa: E402

groups = {
    "layernorm": [(f"{e['arch_id']} wd={e['wd']}", run.gd_dir(e["arch_id"], 0.02, e["wd"])) for e in run.LAYERNORM_RUNS],
    "ntk": [(f"full_200 wd={e['wd']}", run.gd_dir("full_200", 0.02, e["wd"])) for e in run.NTK_RUNS],
    "adamw": [(f"full_200 wd={e['wd']}", run.adamw_dir("full_200", run.ADAMW_LR, e["wd"], 5000)) for e in run.ADAMW_RUNS],
}
for group, entries in groups.items():
    done = sum(run.finished(d) for _, d in entries)
    print(f"{group}: {done}/{len(entries)} finished")
    for label, d in entries:
        state = "done" if run.finished(d) else ("in progress / partial" if os.path.isdir(d) else "not started")
        print(f"    {label:28s} {state}")
conv = f"{run.gd_dir('full_200', 0.02, run.NTK_RUNS[0]['wd'])}/ntk_convergence.pt"
print(f"ntk_convergence: {'done' if os.path.exists(conv) else 'not done'}")
