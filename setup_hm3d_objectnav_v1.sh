mkdir -p data/datasets/objectnav/hm3d
mkdir -p data/scene_datasets

try_symlink() {
    if [ ! -d $2 ] || [ ! -f $2 ]
    then
        ln -s $1 $2
    else
        echo "Error! The following already exists:" ${2}
    fi
}

try_symlink ${ONAV_V1_SCENES} data/scene_datasets/hm3d
try_symlink ${ONAV_V1_EPISODES} data/datasets/objectnav/hm3d/v1

# ONAV_V1_SCENES should contain things like:
#    example
#    hm3d_annotated_basis.scene_dataset_config.json
#    hm3d_basis.scene_dataset_config.json
#    minival
#    train
#    val

# ONAV_V1_EPISODES should contain things like:
#    overfitting
#    s_path_exclude
#    train
#    train_aug
#    train_sample_4k
#    train_sample_4k_unseen
#    train_sample_8k
#    val
#    val_failure
#    val_mini
#    val_remapped
#    val_sample
