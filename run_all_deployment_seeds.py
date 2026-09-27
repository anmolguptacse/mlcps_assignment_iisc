import os, sys, time, subprocess

seeds = [42, 123, 2026, 3407, 7777]
print("=" * 80)
print(f"LAUNCHING 5 PARALLEL SEED DEPLOYMENT WORKERS: {seeds}")
print("=" * 80)

os.makedirs("results/search_campaign/final_seed_models", exist_ok=True)
os.makedirs("scratch/logs", exist_ok=True)

processes = {}
for s in seeds:
    log_path = f"scratch/logs/deployment_seed_{s}.log"
    cmd = [
        "/data2/home/budelalokesh/med_env/bin/python",
        "scratch/build_seed_deployment_member.py",
        "--seed", str(s)
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    
    log_file = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    processes[s] = (p, log_file, log_path)
    print(f"Launched Seed {s:>4} with PID {p.pid} -> Log: {log_path}")

print("\nAll 5 seed workers launched successfully. Monitoring progress...")

# Poll until all complete
start_time = time.time()
completed = set()

while len(completed) < len(seeds):
    time.sleep(15)
    for s in seeds:
        if s not in completed:
            p, log_file, log_path = processes[s]
            ret = p.poll()
            if ret is not None:
                completed.add(s)
                log_file.close()
                elapsed = time.time() - start_time
                if ret == 0:
                    print(f"[{elapsed:6.1f}s] Seed {s:>4} FINISHED SUCCESSFULLY (exit code 0)")
                else:
                    print(f"[{elapsed:6.1f}s] Seed {s:>4} FAILED with exit code {ret}!")

print("=" * 80)
print(f"ALL 5 SEED DEPLOYMENT WORKERS COMPLETED IN {time.time() - start_time:.1f}s!")
print("=" * 80)
