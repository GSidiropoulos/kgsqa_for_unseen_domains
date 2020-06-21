cd ..

cd data

mkdir zero_qg
cd zero_qg

wget https://raw.githubusercontent.com/hadyelsahar/Zeroshot-QuestionGeneration/master/data/copy/entity.vocab

wget https://raw.githubusercontent.com/hadyelsahar/Zeroshot-QuestionGeneration/master/data/copy/property.vocab

wget https://raw.githubusercontent.com/hadyelsahar/Zeroshot-QuestionGeneration/master/data/copy/word.vocab

cd ..

mkdir DAWT
cd DAWT
mkdir part00
mkdir part01
mkdir part02
mkdir part03

# key to decrypt data
wget https://raw.githubusercontent.com/klout/opendata/master/wiki_annotation/v1.gpg_key
for i in {0..3}

do
# download first part of DAWT
wget http://opendata.klout.com/wiki/wiki_annotation/v1/wiki_annotations_json_en_part_0$i.tar.gz.gpg

# decrypt data
gpg --cipher-algo AES256 --passphrase $(cat v1.gpg_key) --output wiki_annotations_json_en_part_0$i.tar.gz --decrypt wiki_annotations_json_en_part_0$i.tar.gz.gpg

# untar
tar xvfz wiki_annotations_json_en_part_0$i.tar.gz

# delete
rm wiki_annotations_json_en_part_0$i.tar.gz
rm wiki_annotations_json_en_part_0$i.tar.gz.gpg

done
