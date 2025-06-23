

with open("data/facebook_combined.txt") as file:
    txt = file.read()

txt = txt.replace(" ", ",").replace("\n", ";")

while txt[-1] == ';':
    txt = txt[:-1]

txt = "1.0|4039|" + txt + "::"

with open("data/facebook_converted.graph", "w") as file:
    file.write(txt)