#nworkers=32 density=1 dnn=resnet50 ./run_cj.sh
#nworkers=32 density=0.001 dnn=resnet50 compressor=topk ./run_cj.sh
#lr=10.0 nworkers=32 density=0.001 dnn=resnet50 compressor=topk ./run_cj.sh
lr=10.0 nworkers=32 density=0.001 dnn=resnet50 compressor=randomk ./run_cj.sh
#lr=16.0 nworkers=32 density=0.001 dnn=resnet50 compressor=gaussian ./run_cj.sh
