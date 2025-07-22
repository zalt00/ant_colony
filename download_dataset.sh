cd data
mkdir other_large_graphs
cd other_large_graphs
wget https://snap.stanford.edu/data/amazon0601.txt.gz
gzip -d amazon0601.txt.gz

wget https://snap.stanford.edu/data/soc-Epinions1.txt.gz
gzip -d soc-Epinions1.txt.gz

wget https://snap.stanford.edu/data/web-NotreDame.txt.gz
gzip -d web-NotreDame.txt.gz

wget https://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip -d wiki-Vote.txt.gz

cd ..

mkdir web-Google
cd    web-Google
wget https://snap.stanford.edu/data/web-Google.txt.gz
gzip -d web-Google.txt.gz
cd..

mkdir soc-LiveJournal1
cd    soc-LiveJournal1
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gzip -d soc-LiveJournal1.txt.gz
cd ..

mkdir soc-pokec-relationships
cd    soc-pokec-relationships
wget https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
gzip -d soc-pokec-relationships.txt.gz
cd ..
