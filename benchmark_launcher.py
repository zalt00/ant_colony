import subprocess

# for graph_name in ["facebook", "enron", "dblp"]:
#     subprocess.call(("cargo", "run" ,"--release", 
#         "--features", "large_graph,mean_path_heuristic", "--", f"clustering_{graph_name}"))

for graph_name in ["facebook", "enron", "dblp"]:
    subprocess.call(("cargo", "run" ,"--release", 
        "--features", "large_graph,mean_path_heuristic,louvain", "--", f"clustering_{graph_name}"))

