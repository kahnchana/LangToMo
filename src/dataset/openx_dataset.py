from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

DATASETS = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    # "language_table",  # Removed temporariliy - need to cache language embeddings (or generate online)
    "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "usc_cloth_sim_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "robo_net",
    # "berkeley_mvp_converted_externally_to_rlds",  # Contains hand-held camera only
    # "berkeley_rpt_converted_externally_to_rlds",  # Contains hand-held camera only
    "kaist_nonprehensile_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    # "dlr_edan_shared_control_converted_externally_to_rlds",  # Empty language annotations
    # "asu_table_top_converted_externally_to_rlds",  # Multiple commands per video (could use later on)
    "stanford_robocook_converted_externally_to_rlds",
    "eth_agent_affordances",
    "imperialcollege_sawyer_wrist_cam",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    # "uiuc_d3field",  # No language annotation
    # "utaustin_mutex",  # Language annotations too long (could use later on)
    "berkeley_fanuc_manipulation",
    "cmu_play_fusion",
    "cmu_stretch",
    "berkeley_gnm_recon",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_sac_son",
]


def print_tree(d, indent=0):
    for key, value in d.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_tree(value, indent + 4)


def dataset2path(dataset_name, root_dir="/nfs/mercedes/hdd1/rt-x"):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    elif dataset_name == "bridge":
        return "/nfs/ws3/hdd1/kanchana/data/bridge/0.1.0"
    else:
        version = "0.1.0"
    return f"{root_dir}/{dataset_name}/{version}"


def as_gif(images, path="temp.gif", duration=100):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


class OpenXDataset:
    def __init__(self, root_dir="/nfs/mercedes/hdd1/rt-x", datasets=DATASETS, split="train"):
        self.dataset_list = datasets
        self.root_dir = root_dir
        self.split = split.split("[")[0]
        self.dataset_dict, self.dataset_sizes = self.init_datasets()

    @staticmethod
    def get_dataset_keys(builder):
        display_key = None
        for option in ["image", "rgb_static", "agentview_rgb", "rgb", "front_rgb", "image_1"]:
            if option in builder.info.features["steps"]["observation"]:
                display_key = option
                break

        lang_options = [
            "language_instruction",
            "structured_language_instruction",
            "natural_language_instruction",
            "instruction",
        ]
        lang_key = None
        in_obs = False

        for option in lang_options:
            if option in builder.info.features["steps"]:
                lang_key = option
                break
            elif option in builder.info.features["steps"]["observation"]:
                lang_key = option
                in_obs = True
                break

        embedding_options = ["natural_language_embedding", "language_embedding"]
        embed_key = None
        for option in embedding_options:
            if option in builder.info.features["steps"]:
                embed_key = option
                assert not in_obs, "mis-match in_obs"
                break
            elif option in builder.info.features["steps"]["observation"]:
                embed_key = option
                assert in_obs, "mis-match in_obs"
                break

        return display_key, lang_key, in_obs, embed_key

    @staticmethod
    def decode_inst_bytes(inst, dataset="language_table", in_obs=False, lang_key="instruction"):
        """Utility to decode encoded language instruction"""
        if in_obs:
            inst = inst["observation"]
        inst = inst[lang_key]
        inst_tensor = tf.convert_to_tensor([inst])
        if dataset == "language_table":
            inst_array = tf.boolean_mask(inst_tensor, inst_tensor != 0)
            return tf.strings.unicode_encode(inst_array, output_encoding="UTF-8")
        else:
            return inst_tensor[0]

    def init_datasets(self):
        dataset_dict = {}
        dataset_sizes = {}
        for dataset in self.dataset_list:
            b = tfds.builder_from_directory(builder_dir=dataset2path(dataset, root_dir=self.root_dir))
            display_key, lang_key, in_obs, embed_key = self.get_dataset_keys(builder=b)
            dataset_sizes[dataset] = b.info.splits[f"{self.split}"].num_examples
            ds = b.as_dataset(split=self.split)

            def episode2steps(episode):
                return episode["steps"]

            def step_map_fn(step):
                decode_func = partial(self.decode_inst_bytes, dataset=dataset, in_obs=in_obs, lang_key=lang_key)
                return {
                    "observation": tf.image.resize(step["observation"][display_key], (128, 128)),
                    "command": decode_func(step),
                    "language_embedding": step["observation"][embed_key] if in_obs else step[embed_key],
                }

            # convert RLDS episode dataset to individual steps & reformat
            ds = ds.map(episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
            ds = ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

            dataset_dict[dataset] = ds

        return dataset_dict, dataset_sizes


if __name__ == "__main__":
    CUR_DATASETS = ["language_table"]
    openx_dataset_instance = OpenXDataset(datasets=CUR_DATASETS, split="train[:10]")  # Load only 10 samples.

    ds = openx_dataset_instance.dataset_dict[CUR_DATASETS[0]]
    step_obs = next(iter(ds))
    print(f"Caption: {step_obs['command'].numpy().decode('utf-8')}")
    print(f"Obs shape: {step_obs['observation'].shape}")
    print(f"Embed shape: {step_obs['language_embedding'].shape}")

    vis_image = Image.fromarray(step_obs["observation"].numpy().astype(np.uint8))
