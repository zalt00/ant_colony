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

for c in [1200.0, 1100.0, 1300.0]:
    for evap in [0.08, 0.07, 0.09]:
        for seed in [1771, 1212]:
            p = subprocess.Popen(f"./target/release/ant_colony.exe test {c} {evap} {seed}", stdout=subprocess.PIPE)
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

