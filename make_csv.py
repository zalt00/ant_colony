from glob import glob
import json
vals = []
for path in glob("result_share/*.json"):
    with open(path) as file:
        data = json.load(file)

    name = path.split("/")[-1].replace("_relabel_shuffle", "").split(".")[0]
    name0 = name
    name = name.replace("soc-", "")
    name = name.replace("-relationships", "").replace("web-", "")

    if "100_" in name:
        n = 100

    elif "_50_" in name:
        n = 50
    elif "1000" in name:
        n = 1000

    elif "amazon" in name:
        n = 403_364

    elif "Epinion" in name:
        n = 75877

    elif "Notre" in name:
        n = 325_729

    elif "wiki" in name:
        n = 7066

    elif "LiveJour" in name:
        n = 4_843_953
    
    elif "pokec" in name:
        n = 1_632_803

    elif "Google" in name:
        n = 855_802

    vals.append((n, name, data, name0))
vals.sort()

with open("result.csv", "w") as file:
    n = 0
    for i in range(0, len(vals), 2):
        nprev = n
        (n, name, data, name0) = vals[i]
        (_, _, data2, _) =       vals[i+1]
        
        if n > nprev:
            if n == 50:
                print("Very small graphs:", file=file)
                print("N;Name;Result (relative to bfs)")
            elif n == 100:
                print("\n\nSmall graphs:", file=file)
                print("N;Name;Result (relative to bfs)")

            elif n == 1000:
                print("\n\nMedium graphs:", file=file)
                print("N;Name;Result (relative to bfs)")

            elif n == 75877:
                print("\n\nLarge graphs:", file=file)
                print("N;Name;Result (relative to bfs)")


        vbest = min(data[0], data2[0])
        print("{};{};{:.4f};%".format(n, name, vbest - 1.0), file=file)


        print(n, name)


