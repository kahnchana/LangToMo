import abc
import dataclasses
import math
from functools import partial
from typing import Any, Dict, Optional, Union

import numpy as np
import reverb
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
import tree
from PIL import Image
from rlds import rlds_types, transformations

from src.dataset import openx_dataset


def _features_to_tensor_spec(feature: tfds.features.FeatureConnector) -> tf.TensorSpec:
    """Converts a tfds Feature into a TensorSpec."""

    def _get_feature_spec(nested_feature: tfds.features.FeatureConnector):
        if isinstance(nested_feature, tf.DType):
            return tf.TensorSpec(shape=(), dtype=nested_feature)
        else:
            return nested_feature.get_tensor_spec()

    # FeaturesDict can sometimes be a plain dictionary, so we use tf.nest to
    # make sure we deal with the nested structure.
    return tf.nest.map_structure(_get_feature_spec, feature)


def _encoded_feature(
    feature: Optional[tfds.features.FeatureConnector],
    image_encoding: Optional[str],
    tensor_encoding: Optional[tfds.features.Encoding],
):
    """Adds encoding to Images and/or Tensors."""

    def _apply_encoding(
        feature: tfds.features.FeatureConnector,
        image_encoding: Optional[str],
        tensor_encoding: Optional[tfds.features.Encoding],
    ):
        if image_encoding and isinstance(feature, tfds.features.Image):
            return tfds.features.Image(
                shape=feature.shape,
                dtype=feature.dtype,
                use_colormap=feature.use_colormap,
                encoding_format=image_encoding,
            )
        if tensor_encoding and isinstance(feature, tfds.features.Tensor) and feature.dtype != tf.string:
            return tfds.features.Tensor(shape=feature.shape, dtype=feature.dtype, encoding=tensor_encoding)
        return feature

    if not feature:
        return None
    return tf.nest.map_structure(lambda x: _apply_encoding(x, image_encoding, tensor_encoding), feature)


@dataclasses.dataclass
class RLDSSpec(metaclass=abc.ABCMeta):
    """Specification of an RLDS Dataset.

    It is used to hold a spec that can be converted into a TFDS DatasetInfo or
    a `tf.data.Dataset` spec.
    """

    observation_info: Optional[tfds.features.FeatureConnector] = None
    action_info: Optional[tfds.features.FeatureConnector] = None
    reward_info: Optional[tfds.features.FeatureConnector] = None
    discount_info: Optional[tfds.features.FeatureConnector] = None
    caption_info: Optional[tfds.features.FeatureConnector] = None
    text_embedding_info: Optional[tfds.features.FeatureConnector] = None
    caption_key: Optional[str] = None
    temb_key: Optional[str] = None
    step_metadata_info: Optional[tfds.features.FeaturesDict] = None
    episode_metadata_info: Optional[tfds.features.FeaturesDict] = None

    def step_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        step = {}
        if self.observation_info:
            step[rlds_types.OBSERVATION] = _features_to_tensor_spec(self.observation_info)
        if self.action_info:
            step[rlds_types.ACTION] = _features_to_tensor_spec(self.action_info)
        if self.discount_info:
            step[rlds_types.DISCOUNT] = _features_to_tensor_spec(self.discount_info)
        if self.reward_info:
            step[rlds_types.REWARD] = _features_to_tensor_spec(self.reward_info)
        if self.caption_info:
            step[self.caption_key] = _features_to_tensor_spec(self.caption_info)
        if self.text_embedding_info:
            step[self.temb_key] = _features_to_tensor_spec(self.text_embedding_info)
        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step[k] = _features_to_tensor_spec(v)

        step[rlds_types.IS_FIRST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_LAST] = tf.TensorSpec(shape=(), dtype=bool)
        step[rlds_types.IS_TERMINAL] = tf.TensorSpec(shape=(), dtype=bool)
        return step

    def episode_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
        """Obtains the TensorSpec of an RLDS step."""
        episode = {}
        episode[rlds_types.STEPS] = tf.data.DatasetSpec(element_spec=self.step_tensor_spec())
        if self.episode_metadata_info:
            for k, v in self.episode_metadata_info.items():
                episode[k] = _features_to_tensor_spec(v)
        return episode

    def to_dataset_config(
        self,
        name: str,
        image_encoding: Optional[str] = None,
        tensor_encoding: Optional[tfds.features.Encoding] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        description: Optional[str] = None,
        overall_description: Optional[str] = None,
    ) -> tfds.rlds.rlds_base.DatasetConfig:
        """Obtains the DatasetConfig for TFDS from the Spec."""
        return tfds.rlds.rlds_base.DatasetConfig(
            name=name,
            description=description,
            overall_description=overall_description,
            homepage=homepage,
            citation=citation,
            observation_info=_encoded_feature(self.observation_info, image_encoding, tensor_encoding),
            action_info=_encoded_feature(self.action_info, image_encoding, tensor_encoding),
            reward_info=_encoded_feature(self.reward_info, image_encoding, tensor_encoding),
            caption_info=_encoded_feature(self.caption_info, image_encoding, tensor_encoding),
            text_embedding_info=_encoded_feature(self.text_embedding_info, image_encoding, tensor_encoding),
            discount_info=_encoded_feature(self.discount_info, image_encoding, tensor_encoding),
            step_metadata_info=_encoded_feature(self.step_metadata_info, image_encoding, tensor_encoding),
            episode_metadata_info=_encoded_feature(self.episode_metadata_info, image_encoding, tensor_encoding),
        )

    def to_features_dict(self):
        """Returns a TFDS FeaturesDict representing the dataset config."""
        step_config = {
            rlds_types.IS_FIRST: tf.bool,
            rlds_types.IS_LAST: tf.bool,
            rlds_types.IS_TERMINAL: tf.bool,
        }

        if self.observation_info:
            step_config[rlds_types.OBSERVATION] = self.observation_info
        if self.action_info:
            step_config[rlds_types.ACTION] = self.action_info
        if self.discount_info:
            step_config[rlds_types.DISCOUNT] = self.discount_info
        if self.reward_info:
            step_config[rlds_types.REWARD] = self.reward_info
        if self.caption_info:
            step_config[self.caption_key] = self.caption_info
        if self.text_embedding_info:
            step_config[self.temb_key] = self.text_embedding_info

        if self.step_metadata_info:
            for k, v in self.step_metadata_info.items():
                step_config[k] = v

        if self.episode_metadata_info:
            return tfds.features.FeaturesDict(
                {
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
                    **self.episode_metadata_info,
                }
            )
        else:
            return tfds.features.FeaturesDict(
                {
                    rlds_types.STEPS: tfds.features.Dataset(step_config),
                }
            )


RLDS_SPEC = RLDSSpec
TENSOR_SPEC = Union[tf.TensorSpec, dict[str, tf.TensorSpec]]


@dataclasses.dataclass
class TrajectoryTransform(metaclass=abc.ABCMeta):
    """Specification the TrajectoryTransform applied to a dataset of episodes.

    A TrajectoryTransform is a set of rules transforming a dataset
    of RLDS episodes to a dataset of trajectories.
    This involves three distinct stages:
    - An optional `episode_to_steps_map_fn(episode)` is called at the episode
      level, and can be used to select or modify steps.
      - Augmentation: an `episode_key` could be propagated to `steps` for
        debugging.
      - Selection: Particular steps can be selected.
      - Stripping: Features can be removed from steps. Prefer using `step_map_fn`.
    - An optional `step_map_fn` is called at the flattened steps dataset for each
      step, and can be used to featurize a step, e.g. add/remove features, or
      augument images
    - A `pattern` leverages DM patterns to set a rule of slicing an episode to a
      dataset of overlapping trajectories.

    Importantly, each TrajectoryTransform must define a `expected_tensor_spec`
    which specifies a nested TensorSpec of the resulting dataset. This is what
    this TrajectoryTransform will produce, and can be used as an interface with
    a neural network.
    """

    episode_dataset_spec: RLDS_SPEC
    episode_to_steps_fn_dataset_spec: RLDS_SPEC
    steps_dataset_spec: Any
    pattern: reverb.structured_writer.Pattern
    episode_to_steps_map_fn: Any
    expected_tensor_spec: TENSOR_SPEC
    step_map_fn: Optional[Any] = None

    def get_for_cached_trajectory_transform(self):
        """Creates a copy of this traj transform to use with caching.

        The returned TrajectoryTransfrom copy will be initialized with the default
        version of the `episode_to_steps_map_fn`, because the effect of that
        function has already been materialized in the cached copy of the dataset.
        Returns:
          trajectory_transform: A copy of the TrajectoryTransform with overridden
            `episode_to_steps_map_fn`.
        """
        traj_copy = dataclasses.replace(self)
        traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
        traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
        return traj_copy

    def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
        """Applies this TrajectoryTransform to the dataset of episodes."""

        # Convert the dataset of episodes to the dataset of steps.
        steps_dataset = episodes_dataset.map(
            self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
        ).flat_map(lambda x: x)

        return self._create_pattern_dataset(steps_dataset)

    def transform_steps_rlds_dataset(self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies this TrajectoryTransform to the dataset of episode steps."""

        return self._create_pattern_dataset(steps_dataset)

    def create_test_dataset(
        self,
    ) -> tf.data.Dataset:
        """Creates a test dataset of trajectories.

        It is guaranteed that the structure of this dataset will be the same as
        when flowing real data. Hence this is a useful construct for tests or
        initialization of JAX models.
        Returns:
          dataset: A test dataset made of zeros structurally identical to the
            target dataset of trajectories.
        """
        zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

        return tf.data.Dataset.from_tensors(zeros)

    def _create_pattern_dataset(self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Create PatternDataset from the `steps_dataset`."""
        config = create_structured_writer_config("temp", self.pattern)

        # Further transform each step if the `step_map_fn` is provided.
        if self.step_map_fn:
            steps_dataset = steps_dataset.map(self.step_map_fn)
        pattern_dataset = reverb.PatternDataset(
            input_dataset=steps_dataset,
            configs=[config],
            respect_episode_boundaries=True,
            is_end_of_episode=lambda x: x[rlds_types.IS_LAST],
        )
        return pattern_dataset


class TrajectoryTransformBuilder(object):
    """Facilitates creation of the `TrajectoryTransform`."""

    def __init__(
        self,
        dataset_spec: RLDS_SPEC,
        episode_to_steps_map_fn=lambda e: e[rlds_types.STEPS],
        step_map_fn=None,
        pattern_fn=None,
        expected_tensor_spec=None,
    ):
        self._rds_dataset_spec = dataset_spec
        self._steps_spec = None
        self._episode_to_steps_map_fn = episode_to_steps_map_fn
        self._step_map_fn = step_map_fn
        self._pattern_fn = pattern_fn
        self._expected_tensor_spec = expected_tensor_spec

    def build(self, validate_expected_tensor_spec: bool = True) -> TrajectoryTransform:
        """Creates `TrajectoryTransform` from a `TrajectoryTransformBuilder`."""

        if validate_expected_tensor_spec and self._expected_tensor_spec is None:
            raise ValueError("`expected_tensor_spec` must be set.")

        episode_ds = zero_episode_dataset_from_spec(self._rds_dataset_spec)

        steps_ds = episode_ds.flat_map(self._episode_to_steps_map_fn)

        episode_to_steps_fn_dataset_spec = self._rds_dataset_spec

        if self._step_map_fn is not None:
            steps_ds = steps_ds.map(self._step_map_fn)

        zeros_spec = transformations.zeros_from_spec(steps_ds.element_spec)  # pytype: disable=wrong-arg-types

        ref_step = reverb.structured_writer.create_reference_step(zeros_spec)

        pattern = self._pattern_fn(ref_step)

        steps_ds_spec = steps_ds.element_spec

        target_tensor_structure = create_reverb_table_signature("temp_table", steps_ds_spec, pattern)

        if validate_expected_tensor_spec and self._expected_tensor_spec != target_tensor_structure:
            raise RuntimeError(
                "The tensor spec of the TrajectoryTransform doesn't "
                "match the expected spec.\n"
                "Expected:\n%s\nActual:\n%s\n"
                % (
                    str(self._expected_tensor_spec).replace("TensorSpec", "tf.TensorSpec"),
                    str(target_tensor_structure).replace("TensorSpec", "tf.TensorSpec"),
                )
            )

        return TrajectoryTransform(
            episode_dataset_spec=self._rds_dataset_spec,
            episode_to_steps_fn_dataset_spec=episode_to_steps_fn_dataset_spec,
            steps_dataset_spec=steps_ds_spec,
            pattern=pattern,
            episode_to_steps_map_fn=self._episode_to_steps_map_fn,
            step_map_fn=self._step_map_fn,
            expected_tensor_spec=target_tensor_structure,
        )


def zero_episode_dataset_from_spec(rlds_spec: RLDS_SPEC):
    """Creates a zero valued dataset of episodes for the given RLDS Spec."""

    def add_steps(episode, step_spec):
        episode[rlds_types.STEPS] = transformations.zero_dataset_like(tf.data.DatasetSpec(step_spec))
        if "fake" in episode:
            del episode["fake"]
        return episode

    episode_without_steps_spec = {k: v for k, v in rlds_spec.episode_tensor_spec().items() if k != rlds_types.STEPS}

    if episode_without_steps_spec:
        episodes_dataset = transformations.zero_dataset_like(tf.data.DatasetSpec(episode_without_steps_spec))
    else:
        episodes_dataset = tf.data.Dataset.from_tensors({"fake": ""})

    episodes_dataset_with_steps = episodes_dataset.map(lambda episode: add_steps(episode, rlds_spec.step_tensor_spec()))
    return episodes_dataset_with_steps


def create_reverb_table_signature(
    table_name: str, steps_dataset_spec, pattern: reverb.structured_writer.Pattern
) -> reverb.reverb_types.SpecNest:
    config = create_structured_writer_config(table_name, pattern)
    reverb_table_spec = reverb.structured_writer.infer_signature([config], steps_dataset_spec)
    return reverb_table_spec


def create_structured_writer_config(table_name: str, pattern: reverb.structured_writer.Pattern) -> Any:
    config = reverb.structured_writer.create_config(pattern=pattern, table=table_name, conditions=[])
    return config


def n_step_pattern_builder(n: int, stride: int = 1) -> Any:
    """Creates trajectory of length `n` from all fields of a `ref_step`."""

    def transform_fn(ref_step):
        traj = {}
        for key in ref_step:
            if isinstance(ref_step[key], dict):
                transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n * stride :: stride], ref_step[key])
                traj[key] = transformed_entry
            else:
                traj[key] = ref_step[key][-n * stride :: stride]

        return traj

    return transform_fn


class DatasetIterator(torch.utils.data.IterableDataset):
    def __init__(self, dataset, length=-1, get_command=True):
        self.get_command = get_command
        self.dataset = dataset
        self.length = length
        self.iterator = iter(dataset)

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        datum = next(self.iterator)
        new_datum = {
            "observation": datum["observation"].numpy(),
            "observation_256": datum["observation_256"].numpy(),
            "caption_embedding": datum["language_embedding"].numpy(),
            # "caption_embedding": tf.math.reduce_mean(datum["language_embedding"], axis=0, keepdims=True).numpy(),
        }
        if self.get_command:
            new_datum["command"] = datum["command"][0].numpy()
        return new_datum

    def __len__(self):
        return self.length


def custom_collate_fn(batch):
    converted_batch = []
    for item in batch:
        # Recursively apply conversion for dictionary items
        new_item = {"observation": item["observation"].numpy(), "command": item["command"][0].numpy()}
        converted_batch.append(new_item)
    return torch.utils.data.default_collate(converted_batch)


class OpenXTrajectoryDataset(openx_dataset.OpenXDataset):
    def __init__(
        self, *init_args, trajectory_length=3, traj_stride=1, img_size=128, infinite_repeat=False, **init_kwargs
    ):
        self.trajectory_length = trajectory_length
        self.trajectory_stride = traj_stride
        self.img_size = img_size
        self.inifinite_repeat = infinite_repeat
        super().__init__(*init_args, **init_kwargs)

    def init_datasets(self):
        dataset_dict = {}
        dataset_sizes = {}
        for dataset in self.dataset_list:
            b = tfds.builder_from_directory(builder_dir=openx_dataset.dataset2path(dataset, root_dir=self.root_dir))
            display_key, lang_key, in_obs, embed_key = self.get_dataset_keys(builder=b)
            ds = b.as_dataset(split=self.split)

            def is_successful(episode):
                return episode["episode_metadata"]["success"]

            if dataset in ["ucsd_pick_and_place_dataset_converted_externally_to_rlds"]:
                ds = ds.filter(is_successful)

            dataset_sizes[dataset] = b.info.splits["train"].num_examples
            decode_func = partial(self.decode_inst_bytes, dataset=dataset, in_obs=in_obs, lang_key=lang_key)

            if in_obs:
                rlds_spec = RLDSSpec(observation_info=b.info.features["steps"]["observation"])
            else:
                rlds_spec = RLDSSpec(
                    observation_info=b.info.features["steps"]["observation"],
                    caption_info=b.info.features["steps"][lang_key],
                    text_embedding_info=b.info.features["steps"][embed_key],
                    caption_key=lang_key,
                    temb_key=embed_key,
                )

            def step_map_fn(step):
                transformed_step = {}
                image = tf.image.resize(step["observation"][display_key], [self.img_size, self.img_size])
                image = tf.cast(image, tf.float32) / 255.0
                image = tf.transpose(image, perm=[2, 0, 1])  # HWC to CHW
                image_256 = tf.image.resize(step["observation"][display_key], [256, 256])
                image_256 = tf.cast(image_256, tf.float32) / 255.0
                image_256 = tf.transpose(image_256, perm=[2, 0, 1])
                transformed_step["observation"] = image
                transformed_step["observation_256"] = image_256
                transformed_step["command"] = decode_func(step)
                transformed_step["language_embedding"] = step["observation"][embed_key] if in_obs else step[embed_key]
                transformed_step["is_first"] = step["is_first"]
                transformed_step["is_last"] = step["is_last"]
                transformed_step["is_terminal"] = step["is_terminal"]
                return transformed_step

            cur_stride = (
                self.trajectory_stride if isinstance(self.trajectory_stride, int) else self.trajectory_stride[dataset]
            )
            pattern_func = n_step_pattern_builder(self.trajectory_length, stride=cur_stride)
            trajectory_transform = TrajectoryTransformBuilder(
                rlds_spec, step_map_fn=step_map_fn, pattern_fn=pattern_func
            ).build(validate_expected_tensor_spec=False)

            trajectory_ds = trajectory_transform.transform_episodic_rlds_dataset(ds)
            trajectory_ds = trajectory_ds.prefetch(tf.data.AUTOTUNE)
            if self.inifinite_repeat:
                trajectory_ds = trajectory_ds.repeat()

            dataset_dict[dataset] = trajectory_ds

        return dataset_dict, dataset_sizes


SELECTED_DS_INFO = [
    {"name": "fractal20220817_data", "robot": "Google Robot", "episodes": 73499, "size": 111.06},
    {"name": "taco_play", "robot": "Franka", "episodes": 3242, "size": 47.77},
    {"name": "language_table", "robot": "xArm", "episodes": 442226, "size": 399.22},
    {"name": "stanford_hydra", "robot": "Franka", "episodes": 550, "size": 72.48},
    {"name": "ucsd_pick_and_place", "robot": "xArm", "episodes": 1355, "size": 3.53},
    {"name": "iamlab_cmu_pickup_insert_", "robot": "Franka", "episodes": 520, "size": 50.29},
    {"name": "utaustin_mutex", "robot": "Franka", "episodes": 1500, "size": 20.79},
]

DS_TO_FPS = {
    "fractal20220817_data": 3,
    "taco_play": 15,
    "language_table": 10,
    "stanford_hydra_dataset_converted_externally_to_rlds": 10,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": 3,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20,
    "utaustin_mutex": 20,
}


def get_ds_weights():
    robot_weights = {"xArm": 3.0, "Franka": 1.5, "Google Robot": 1.0}

    def compute_score(entry):
        robot_weight = robot_weights.get(entry["robot"], 1.0)
        sqrt_episodes = math.sqrt(entry["episodes"])
        return robot_weight * sqrt_episodes / entry["size"]

    raw_scores = [compute_score(d) for d in SELECTED_DS_INFO]

    # Normalize
    total = sum(raw_scores)
    ds_weights = [score / total for score in raw_scores]

    return ds_weights


if __name__ == "__main__":
    dataset_names = [
        "fractal20220817_data",
        "taco_play",
        "language_table",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "utaustin_mutex",
    ]

    dataset_to_stride = {x: int(y // 3) for x, y in DS_TO_FPS.items()}

    traj_dataset = OpenXTrajectoryDataset(
        datasets=dataset_names,
        split="train[:10]",
        trajectory_length=9,
        traj_stride=dataset_to_stride,
        img_size=256,
        root_dir="/nfs/ws2/kanchana/openx",
    )

    mixed_dataset = tf.data.experimental.sample_from_datasets(
        list(traj_dataset.dataset_dict.values()), weights=[1 / len(dataset_names)] * len(dataset_names)
    )

    if False:
        vis_dict = {}
        for ds_name in tqdm.tqdm(dataset_names):
            cur_ds = iter(traj_dataset.dataset_dict[ds_name])
            trajectory = next(cur_ds)
            vis_trajectory = np.transpose(trajectory["observation"], axes=(0, 2, 3, 1))
            image_list = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]
            vis_dict[ds_name] = image_list

    if False:
        cur_ds = iter(traj_dataset.dataset_dict[dataset_names[0]])
        trajectory = next(cur_ds)
        vis_trajectory = np.transpose(trajectory["observation"], axes=(0, 2, 3, 1))
        image_list = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]

        from IPython import display

        display.Image(openx_dataset.as_gif([x.resize((512, 512)) for x in image_list]))

    if False:
        train_dataset = DatasetIterator(traj_dataset.dataset_dict[dataset_names[0]], get_command=False)
        cur_dataset = DatasetIterator(traj_dataset.dataset_dict[dataset_names[0]])  # include command for val (default)
        # cur_dataset = DatasetIterator(mixed_dataset)
        train_loader = torch.utils.data.DataLoader(cur_dataset, sampler=None, batch_size=4)
        for batch in train_loader:
            print(batch["observation"].shape)
            print(len(batch["command"]))
            print(batch["caption_embedding"].shape)
            break
