mkdir data
cd data

echo word2vec embeddings
# download word2vec
mkdir embeddings
cd embeddings

wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

gunzip GoogleNews-vectors-negative300.bin.gz
cd ..

echo Download SimpleQuestions
# download SimpleQuestions dataset
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz
tar -xvzf SimpleQuestions_v2.tgz
rm SimpleQuestions_v2.tgz

echo Download necessary data
# 
wget -O fb_id2sentences_idstr_label_type.pickle "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=fb_id2sentences_idstr_label_type.pickle"
wget -O inverted_index.pkl "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=inverted_index.pkl"
wget -O freebase_id_wiki_url.en_fixed "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=freebase_id_wiki_url.en_fixed"
wget -O mid2entity_list_simple.pkl "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=mid2entity_list_simple.pkl"
wget -O entity2mid_list_simple.pkl "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=entity2mid_list_simple.pkl"
wget -O mid2ent.pkl "https://surfdrive.surf.nl/files/index.php/s/hiNeHPTuFZ3HtEp/download?path=%2Fkgsqa_for_unseen_domains&files=mid2ent.pkl"

