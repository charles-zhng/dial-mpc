from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np
import jax
import jax.numpy as jp
from functools import partial
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model
import dial_mpc.envs as dial_envs

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig

_MOCAP_HZ = 50
_JOINT_NAMES = [
    "vertebra_1_extend",
    "hip_L_supinate",
    "hip_L_abduct",
    "hip_L_extend",
    "knee_L",
    "ankle_L",
    "toe_L",
    "hip_R_supinate",
    "hip_R_abduct",
    "hip_R_extend",
    "knee_R",
    "ankle_R",
    "toe_R",
    "vertebra_C11_extend",
    "vertebra_cervical_1_bend",
    "vertebra_axis_twist",
    "atlas",
    "mandible",
    "scapula_L_supinate",
    "scapula_L_abduct",
    "scapula_L_extend",
    "shoulder_L",
    "shoulder_sup_L",
    "elbow_L",
    "wrist_L",
    "scapula_R_supinate",
    "scapula_R_abduct",
    "scapula_R_extend",
    "shoulder_R",
    "shoulder_sup_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
]
_BODY_NAMES = [
    "torso",
    "pelvis",
    "upper_leg_L",
    "lower_leg_L",
    "foot_L",
    "upper_leg_R",
    "lower_leg_R",
    "foot_R",
    "skull",
    "jaw",
    "scapula_L",
    "upper_arm_L",
    "lower_arm_L",
    "finger_L",
    "scapula_R",
    "upper_arm_R",
    "lower_arm_R",
    "finger_R",
]

_END_EFF_NAMES = [
    "foot_L",
    "foot_R",
    "hand_L",
    "hand_R",
    "skull",
]

import os
from preprocess import process_clip_to_train, ReferenceClip
import pickle

clip_id = 84  # 84 is the walking in half circle one
reference_path = f"clips/{clip_id}.p"

if not os.path.exists(reference_path):
    os.makedirs(os.path.dirname(reference_path), exist_ok=True)

    # Process rodent clip and save as pickle
    reference_clip = process_clip_to_train(
        stac_path="./transform_snips_new.p",
        start_step=clip_id * 250,
        clip_length=250,
        mjcf_path="../models/rodent/rodent.xml",
    )
    with open(reference_path, "wb") as file:
        # Use pickle.dump() to save the data to the file
        pickle.dump(reference_clip, file)
else:
    with open(reference_path, "rb") as file:
        # Use pickle.load() to load the data from the file
        reference_clip = pickle.load(file)


@dataclass
class RodentImitationConfig(BaseEnvConfig):
    reference_clip: ReferenceClip = reference_clip
    mocap_hz: int = 50
    clip_len: int = 250
    scale_factor: float = 0.9
    ref_len: int = 5
    too_far_dist = 0.1
    bad_pose_dist = jp.inf
    bad_quat_dist = jp.inf
    ctrl_cost_weight = 0.01
    pos_reward_weight = 1.0
    quat_reward_weight = 1.0
    joint_reward_weight = 1.0
    bodypos_reward_weight = 1.0
    endeff_reward_weight = 1.0
    healthy_z_range = (0.0325, 0.5)
    reset_noise_scale = 1e-3
    solver = "cg"
    iterations: int = 6
    ls_iterations: int = 6


class RodentImitation(BaseEnv):
    def __init__(self, config: RodentImitationConfig):
        super().__init__(config)

        # custom initializations below...
        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mju_str2Type("body"), "torso"
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(
                    self.sys.mj_model, mujoco.mju_str2Type("joint"), joint
                )
                for joint in _JOINT_NAMES
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(self.sys.mj_model, mujoco.mju_str2Type("body"), body)
                for body in _BODY_NAMES
            ]
        )

        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(self.sys.mj_model, mujoco.mju_str2Type("body"), body)
                for body in _END_EFF_NAMES
            ]
        )

        self._framerate = config.mocap_hz
        self._clip_len = config.clip_len
        self.reference_clip = config.reference_clip
        self._reset_noise_scale = config.reset_noise_scale
        self._ref_len = config.ref_len

        self._pos_reward_weight = config.pos_reward_weight
        self._quat_reward_weight = config.quat_reward_weight
        self._joint_reward_weight = config.joint_reward_weight
        self._bodypos_reward_weight = config.bodypos_reward_weight
        self._endeff_reward_weight = config.endeff_reward_weight
        self._bad_pose_dist = config.bad_pose_dist
        self._too_far_dist = config.too_far_dist
        self._bad_quat_dist = config.bad_quat_dist
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._healthy_z_range = config.healthy_z_range

    def make_system(self, config: RodentImitationConfig) -> System:
        model_path = "../models/rodent/rodent.xml"
        # scale rodent model hack
        root = mjcf_dm.from_path(model_path)
        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        for actuator in root.find_all("actuator"):
            actuator.gainprm = [actuator.forcerange[1]]
            del actuator.biastype
            del actuator.biasprm
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        # TODO: implement reset
        rng, key1, key2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        qpos_with_pos = (
            jp.array(self.sys.qpos0).at[:2].set(self.reference_clip.position[0][:2])
        )

        # Add quat
        new_qpos = qpos_with_pos.at[3:7].set(self.reference_clip.quaternion[0])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            key1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(key2, (self.sys.nv,), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "step": 0,
        }

        obs = self._get_obs(pipeline_state, 0)
        reward, done = jp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # TODO: implement step
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        state.info["step"] += 1
        # Logic for getting current frame aligned with simulation time
        cur_frame = jp.array(data.time * self._framerate, int) % self._clip_len

        pos_distance = data.qpos[:3] - self.reference_clip.position[cur_frame]
        pos_reward = self._pos_reward_weight * jp.exp(-400 * jp.sum(pos_distance) ** 2)

        quat_distance = jp.sum(
            self._bounded_quat_dist(
                data.qpos[3:7], self.reference_clip.quaternion[cur_frame]
            )
            ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)

        joint_distance = (
            jp.sum(data.qpos[7:] - self.reference_clip.joints[cur_frame]) ** 2
        )
        joint_reward = self._joint_reward_weight * jp.exp(-0.5 * joint_distance)

        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -6.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - self.reference_clip.body_positions[cur_frame][self._body_idxs]
                ).flatten()
            )
            ** 2
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -400
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - self.reference_clip.body_positions[cur_frame][self._endeff_idxs]
                ).flatten()
            )
            ** 2
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx][2] > max_z, 0.0, is_healthy)

        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, cur_frame)
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + bodypos_reward
            + endeff_reward
            - ctrl_cost
        )
        done = 1.0 - is_healthy
        done = jp.max(jp.array([done, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_quadctrl=-ctrl_cost,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=1 - is_healthy,
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, cur_frame) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_len,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self.reference_clip)

        track_pos_local = jax.vmap(lambda a, b: math.rotate(a, b), in_axes=(0, None))(
            ref_traj.position - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(lambda a, b: math.relative_quat(a, b), in_axes=(None, 0))(
            data.qpos[3:7],
            ref_traj.quaternion,
        ).flatten()

        joint_dist = (ref_traj.joints - data.qpos[7:])[:, self._joint_idxs].flatten()

        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        return jp.concatenate(
            [
                data.ctrl,
                data.qpos,
                data.qvel,
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]


brax_envs.register_environment("rodent_imitation", RodentImitation)
dial_envs.register_config("rodent_imitation", RodentImitationConfig)
