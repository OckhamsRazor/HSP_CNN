rm -rf test_data/*
[ -d HL_output/time ] || mkdir HL_output/time
[ -d HL_output/score ] || mkdir HL_output/score
[ -d HL_output/audio ] || mkdir HL_output/audio

python2.7 HL30.pyc
python mel_extract.py -i HL_output/audio -o mels
mv mels/*.npy test_data/xte.npy
python extract_feats.py
python make_ex_data.py
python standardize.py
python test_model.py
python3 cnn.ret30.fc.py -i test_data/ -ps -o ret30 -tg

rm -rf jy_feat/* ex_data/* std/* HL_output/* mels/
