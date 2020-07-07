dataset:
	python3 -m synsight.make_dataset \
        --in /dataset/vggface2/orig/ \
        --splits train,test \
        --out /dataset/vggface2/64/ \
        --size 64
