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
for i in range(0, 16*step, step):
    p = subprocess.Popen(f"./target/release/ant_colony.exe test {i} {i+step}", stdout=subprocess.PIPE)
    proc.append(p)

for p in proc:
    (result, _) = p.communicate()
    (t, ares, gres, gsres) = result.split(b";")
    print(float(t), float(ares), float(gres), float(gsres))
    sua += float(ares)
    sug += float(gres)
    sugs += float(gsres)
    c+= 1
    p.wait()


print(sua / c, sug / c)

print(time.perf_counter() - t1)

