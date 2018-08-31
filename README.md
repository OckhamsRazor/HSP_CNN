# HSP_CNN
A CNN model for hit song prediction (HSP) in Lang-Chi Yu, Yi-Hsuan Yang, Yun-Ning Hung, and Yi-An Chen, “Hit Song Prediction for Pop Music by Siamese CNN with Ranking Loss,” arXiv preprint arXiv:1710.10814 (2017).

https://arxiv.org/abs/1710.10814

# USAGE
- Put mp3 files (songs) for test in input/
- ./run.sh
- ret30.npy will be produced. It contains retention-30 values (a song popularity metric) and embeddings of input songs.

Please refer to run.sh for more information. Model training, testing parts can be found in cnn.ret30.fc.py.
