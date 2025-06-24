import subprocess
import time

subprocess.run("cargo build --release")

t1 = time.perf_counter()
step = 3
sua = 0.0
sug = 0.0
sugs = 0.0
c = 0
proc = []

for c in [8000.0]:
    for evap in [0.4]:
        for seed in [1771, 1212]:
            for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:   
                p = subprocess.Popen(f"./target/release/ant_colony.exe test {c} {evap} {seed} {w}", stdout=subprocess.PIPE)
                proc.append(p)

# for c in [800.0, 1000.0, 600.0, 1200.0]:
#     for evap in [0.08, 0.1, 0.12, 0.15]:
#         p = subprocess.Popen(f"./target/release/ant_colony.exe test {c} {evap}", stdout=subprocess.PIPE)
#         proc.append(p)

for p in proc:
    (result, _) = p.communicate()
    print(result.decode("utf-8"))
    p.wait()


print(time.perf_counter() - t1)

