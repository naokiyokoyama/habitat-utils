mkdir -p data/datasets/objectnav/hm3d/v1
mkdir -p data/scene_datasets/hm3d
ln -s /coc/testnvme/nyokoyama3/flash_datasets/hm3d_v0.2/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
ln -s /coc/testnvme/nyokoyama3/flash_datasets/hm3d_v0.2/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
ln -s /coc/testnvme/datasets/habitat-sim-datasets/pointnav data/datasets/pointnav

# Scene datasets
ln -s /coc/testnvme/nyokoyama3/flash_datasets/hm3d_v0.2/hm3d-train-habitat-v0.2 data/scene_datasets/hm3d/train
ln -s /coc/testnvme/nyokoyama3/flash_datasets/hm3d_v0.2/hm3d-val-habitat-v0.2 data/scene_datasets/hm3d/val
ln -s /coc/testnvme/datasets/habitat-sim-datasets/gibson_train_val data/scene_datasets/gibson
