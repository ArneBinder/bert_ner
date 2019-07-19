create conda env
cpu: `conda create -n aprobe python spacy ftfy scikit-learn pydot pytorch pytorch-pretrained-bert ignite tensorboardx=1.6 tensorflow keras blas=*=mkl -c pytorch -c conda-forge`
gpu: `conda create -n aprobe python spacy ftfy scikit-learn pydot pytorch pytorch-pretrained-bert ignite tensorboardx=1.6 tensorflow-gpu keras blas=*=mkl -c pytorch -c conda-forge`

train conll2003 DEPRECATED
`CUDA_VISIBLE_DEVICES=0 python train.py --logdir checkpoints/feature --batch_size 128 --top_rnns --lr 1e-4 --n_epochs 30 --trainset /home/binder/corpora/conll2003/train.txt --validset /home/binder/corpora/conll2003/valid.txt`


train conll2003 via keras @gpu8
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_keras.py --batch_size 1024 --top_rnns --lr 1e-4 --n_epochs 30 --use_default_tagset --trainset /home/binder/corpora/conll2003/train.txt --validset /home/binder/corpora/conll2003/valid.txt &> train0.log`


`CUDA_VISIBLE_DEVICES=3,4,5,6 python train_keras.py --batch_size 128 --top_rnns --lr 1e-4 --n_epochs 30 --use_default_tagset --trainset /home/binder/corpora/conll2003/train.txt --validset /home/binder/corpora/conll2003/valid.txt &> train3.log`