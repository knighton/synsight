train:
	python3 -m synsight.train \
        --samples /dataset/vggface2/48/samples.npy \
        --size 48

dataset:
	python3 -m synsight.make_dataset \
        --in /dataset/vggface2/orig/ \
        --splits train,test \
        --out /dataset/vggface2/48/ \
        --size 48
