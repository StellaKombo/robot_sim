"""Phase-A validation for the shared rigid coordinate frame."""

from pathlib import Path

import numpy as np
import pytest

from utils.relative_frame import RelativeFrame


# Samples from data/data/sync_lissa_kcf02/g1_23dof_data.npz.  Keeping the
# small fixture here makes the geometry tests independent of generated logs.
MUJOCO_BASE_POSES = np.array(
    [
        [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0],
        [0.8217391, -0.7760436, 0.7830768,
         -0.07737438, 0.00321054, -0.007874368, -0.9969658],
        [-1.3696941, -0.537736, 0.785629,
         0.87205696, 0.01208241, -0.000583725, -0.48925492],
        [1.5099434, 0.16684623, 0.7789732,
         0.77889866, 0.003258298, -0.001122839, -0.62714034],
        [-1.1413283, 0.694811, 0.7885143,
         0.1402713, 0.019683331, 0.023580262, -0.98963654],
        [0.50017756, 0.49803656, 0.7849658,
         0.94581324, 0.01601389, 0.008239923, 0.32421136],
    ],
    dtype=np.float64,
)

MUJOCO_BASE_VELOCITIES = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.2551776, 0.20947993, 0.005734099],
        [0.03189802, -0.24688411, 0.004326369],
        [0.12106148, -0.27465564, 0.013657231],
        [-0.29031897, 0.068835355, 0.027777396],
        [0.3573348, 0.32016286, 0.016578829],
    ],
    dtype=np.float64,
)


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def normalized(quaternions):
    return quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)


def assert_same_orientation(actual, expected, atol=2e-12):
    """Compare orientations while allowing the equivalent q/-q representation."""
    actual = normalized(np.asarray(actual))
    expected = normalized(np.asarray(expected))
    alignment = np.abs(np.sum(actual * expected, axis=-1))
    np.testing.assert_allclose(alignment, 1.0, rtol=0.0, atol=atol)


def test_known_yaw_and_translation_transform():
    frame = RelativeFrame([1.0, 2.0, 3.0], yaw_quaternion(np.pi / 2.0))

    np.testing.assert_allclose(
        frame.position_to_local([1.0, 3.0, 3.0]),
        [1.0, 0.0, 0.0],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        frame.linear_velocity_to_local([0.0, 2.0, 0.0]),
        [2.0, 0.0, 0.0],
        atol=2e-15,
    )
    assert_same_orientation(
        frame.orientation_to_local(yaw_quaternion(np.pi)),
        yaw_quaternion(np.pi / 2.0),
    )


def test_real_mujoco_samples_round_trip_in_one_fixed_frame():
    frame = RelativeFrame.from_pose(
        MUJOCO_BASE_POSES[1, :3], MUJOCO_BASE_POSES[1, 3:7]
    )

    position_local = frame.position_to_local(MUJOCO_BASE_POSES[:, :3])
    orientation_local = frame.orientation_to_local(MUJOCO_BASE_POSES[:, 3:7])
    velocity_local = frame.linear_velocity_to_local(MUJOCO_BASE_VELOCITIES)

    np.testing.assert_allclose(
        frame.position_to_world(position_local),
        MUJOCO_BASE_POSES[:, :3],
        rtol=0.0,
        atol=2e-12,
    )
    assert_same_orientation(
        frame.orientation_to_world(orientation_local),
        MUJOCO_BASE_POSES[:, 3:7],
    )
    np.testing.assert_allclose(
        frame.linear_velocity_to_world(velocity_local),
        MUJOCO_BASE_VELOCITIES,
        rtol=0.0,
        atol=2e-12,
    )


def test_arbitrary_leading_batch_dimensions_are_preserved():
    frame = RelativeFrame.from_pose(
        MUJOCO_BASE_POSES[3, :3], MUJOCO_BASE_POSES[3, 3:7]
    )
    positions = MUJOCO_BASE_POSES[:, :3].reshape(2, 3, 3)
    orientations = MUJOCO_BASE_POSES[:, 3:7].reshape(2, 3, 4)
    velocities = MUJOCO_BASE_VELOCITIES.reshape(2, 3, 3)

    assert frame.position_to_local(positions).shape == (2, 3, 3)
    assert frame.orientation_to_local(orientations).shape == (2, 3, 4)
    assert frame.linear_velocity_to_local(velocities).shape == (2, 3, 3)
    assert frame.orientation_heading_to_local_sincos(orientations).shape == (
        2, 3, 2
    )


def test_local_values_are_invariant_to_global_translation_and_yaw():
    pose = MUJOCO_BASE_POSES
    velocity = MUJOCO_BASE_VELOCITIES
    frame = RelativeFrame.from_pose(pose[2, :3], pose[2, 3:7])

    global_yaw = 1.17
    cos_yaw, sin_yaw = np.cos(global_yaw), np.sin(global_yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0],
         [sin_yaw, cos_yaw, 0.0],
         [0.0, 0.0, 1.0]]
    )
    translation = np.array([4.2, -3.7, 1.3])
    global_quaternion = yaw_quaternion(global_yaw)

    transformed_position = pose[:, :3] @ rotation.T + translation
    transformed_velocity = velocity @ rotation.T
    transformed_orientation = np.empty_like(pose[:, 3:7])
    gw, gx, gy, gz = global_quaternion
    qw, qx, qy, qz = pose[:, 3:7].T
    transformed_orientation[:, 0] = gw * qw - gx * qx - gy * qy - gz * qz
    transformed_orientation[:, 1] = gw * qx + gx * qw + gy * qz - gz * qy
    transformed_orientation[:, 2] = gw * qy - gx * qz + gy * qw + gz * qx
    transformed_orientation[:, 3] = gw * qz + gx * qy - gy * qx + gz * qw

    transformed_frame = RelativeFrame.from_pose(
        transformed_position[2], transformed_orientation[2]
    )

    np.testing.assert_allclose(
        transformed_frame.position_to_local(transformed_position),
        frame.position_to_local(pose[:, :3]),
        rtol=0.0,
        atol=3e-12,
    )
    assert_same_orientation(
        transformed_frame.orientation_to_local(transformed_orientation),
        frame.orientation_to_local(pose[:, 3:7]),
    )
    np.testing.assert_allclose(
        transformed_frame.linear_velocity_to_local(transformed_velocity),
        frame.linear_velocity_to_local(velocity),
        rtol=0.0,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        transformed_frame.orientation_heading_to_local_sincos(
            transformed_orientation
        ),
        frame.orientation_heading_to_local_sincos(pose[:, 3:7]),
        rtol=0.0,
        atol=3e-12,
    )


def test_heading_sincos_round_trip_is_modulo_two_pi():
    frame = RelativeFrame.from_pose(
        [2.0, -1.0, 0.8], yaw_quaternion(0.7), yaw_only=True
    )
    headings = np.array([0.7, 0.7 + np.pi / 2.0, 0.7 + 2.0 * np.pi, -8.4])

    encoded = frame.heading_to_local_sincos(headings)
    decoded = frame.heading_sincos_to_world(encoded)
    expected = np.arctan2(np.sin(headings), np.cos(headings))

    np.testing.assert_allclose(decoded, expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        np.linalg.norm(encoded, axis=-1), 1.0, rtol=0.0, atol=2e-15
    )


def test_yaw_only_factory_keeps_world_vertical():
    tilted_pose = MUJOCO_BASE_POSES[4]
    frame = RelativeFrame.from_pose(
        tilted_pose[:3], tilted_pose[3:7], yaw_only=True
    )

    np.testing.assert_allclose(frame.orientation_world[1:3], 0.0, atol=2e-15)
    np.testing.assert_allclose(
        frame.linear_velocity_to_local([0.0, 0.0, 1.0]),
        [0.0, 0.0, 1.0],
        atol=2e-15,
    )


def test_saved_trajectory_windows_use_independent_fixed_ground_frames():
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data/data/sync_lissa_kcf02/g1_23dof_data.npz"
    )
    if not data_path.is_file():
        pytest.skip("saved humanoid trajectory is not available")

    with np.load(data_path, allow_pickle=False) as archive:
        poses = np.asarray(archive["q_log"][0], dtype=np.float64)
        velocities = np.asarray(archive["v_log"][0, :, :3], dtype=np.float64)

    for start in (0, 100, 500):
        stop = start + 200
        window_pose = poses[start:stop]
        window_velocity = velocities[start:stop]
        ground_origin = np.array(
            [window_pose[0, 0], window_pose[0, 1], 0.0]
        )
        frame = RelativeFrame.from_pose(
            ground_origin, window_pose[0, 3:7], yaw_only=True
        )
        saved_origin = frame.origin_world.copy()
        saved_orientation = frame.orientation_world.copy()

        position_local = frame.position_to_local(window_pose[:, :3])
        orientation_local = frame.orientation_to_local(window_pose[:, 3:7])
        velocity_local = frame.linear_velocity_to_local(window_velocity)
        heading_local = frame.orientation_heading_to_local_sincos(
            window_pose[:, 3:7]
        )

        np.testing.assert_allclose(position_local[0, :2], 0.0, atol=2e-14)
        np.testing.assert_allclose(
            position_local[0, 2], window_pose[0, 2], atol=2e-14
        )
        np.testing.assert_allclose(heading_local[0], [0.0, 1.0], atol=2e-14)
        np.testing.assert_allclose(
            frame.position_to_world(position_local),
            window_pose[:, :3],
            rtol=0.0,
            atol=2e-12,
        )
        assert_same_orientation(
            frame.orientation_to_world(orientation_local),
            window_pose[:, 3:7],
        )
        np.testing.assert_allclose(
            frame.linear_velocity_to_world(velocity_local),
            window_velocity,
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_array_equal(frame.origin_world, saved_origin)
        np.testing.assert_array_equal(
            frame.orientation_world, saved_orientation
        )


@pytest.mark.parametrize(
    "origin, orientation, message",
    [
        ([0.0, 0.0], [1.0, 0.0, 0.0, 0.0], "origin_world"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "zero quaternion"),
        ([0.0, np.nan, 0.0], [1.0, 0.0, 0.0, 0.0], "finite"),
    ],
)
def test_invalid_frame_is_rejected(origin, orientation, message):
    with pytest.raises(ValueError, match=message):
        RelativeFrame(origin, orientation)


def test_frame_definition_is_immutable():
    origin = np.array([1.0, 2.0, 3.0])
    orientation = yaw_quaternion(0.4)
    frame = RelativeFrame(origin, orientation)
    origin[:] = 0.0
    orientation[:] = 0.0

    np.testing.assert_allclose(frame.origin_world, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        frame.origin_world[0] = 5.0
