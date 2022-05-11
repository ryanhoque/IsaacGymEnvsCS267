#include <torch/extension.h>
#include <vector>

void calculate_cartpole_reward(torch::Tensor pole_angle,
    torch::Tensor pole_vel,
	torch::Tensor cart_vel,
    torch::Tensor cart_pos,
    float reset_dist,
    torch::Tensor reset_buf,
	torch::Tensor progress_buf,
	float max_episode_length,
    torch::Tensor reward_buf);

void calculate_frankacabinet_reward(
    torch::Tensor reset_buf,
    torch::Tensor progress_buf,
    torch::Tensor actions,
    torch::Tensor cabinet_dof_pos,
    torch::Tensor franka_grasp_pos,
    torch::Tensor drawer_grasp_pos,
    torch::Tensor franka_grasp_rot,
    torch::Tensor drawer_grasp_rot,
    torch::Tensor franka_lfinger_pos,
    torch::Tensor franka_rfinger_pos,
    torch::Tensor gripper_forward_axis,
    torch::Tensor drawer_inward_axis,
    torch::Tensor gripper_up_axis,
    torch::Tensor drawer_up_axis,
    int num_envs,
    float dist_reward_scale,
    float rot_reward_scale,
    float around_handle_reward_scale,
    float open_reward_scale,
    float finger_dist_reward_scale,
    float action_penalty_scale,
    float distX_offset,
    float max_episode_length,
    torch::Tensor reward_buf
);

void calculate_franka_reward_parallel(
    torch::Tensor reset_buf,
    torch::Tensor progress_buf,
    torch::Tensor actions,
    torch::Tensor cabinet_dof_pos,
    torch::Tensor franka_grasp_pos,
    torch::Tensor drawer_grasp_pos,
    torch::Tensor franka_grasp_rot,
    torch::Tensor drawer_grasp_rot,
    torch::Tensor franka_lfinger_pos,
    torch::Tensor franka_rfinger_pos,
    torch::Tensor gripper_forward_axis,
    torch::Tensor drawer_inward_axis,
    torch::Tensor gripper_up_axis,
    torch::Tensor drawer_up_axis,
    int num_envs,
    float dist_reward_scale,
    float rot_reward_scale,
    float around_handle_reward_scale,
    float open_reward_scale,
    float finger_dist_reward_scale,
    float action_penalty_scale,
    float distX_offset,
    float max_episode_length,
    torch::Tensor reward_buf
);

void calculate_franka_reward_parallel_stream(
    torch::Tensor reset_buf,
    torch::Tensor progress_buf,
    torch::Tensor actions,
    torch::Tensor cabinet_dof_pos,
    torch::Tensor franka_grasp_pos,
    torch::Tensor drawer_grasp_pos,
    torch::Tensor franka_grasp_rot,
    torch::Tensor drawer_grasp_rot,
    torch::Tensor franka_lfinger_pos,
    torch::Tensor franka_rfinger_pos,
    torch::Tensor gripper_forward_axis,
    torch::Tensor drawer_inward_axis,
    torch::Tensor gripper_up_axis,
    torch::Tensor drawer_up_axis,
    int num_envs,
    float dist_reward_scale,
    float rot_reward_scale,
    float around_handle_reward_scale,
    float open_reward_scale,
    float finger_dist_reward_scale,
    float action_penalty_scale,
    float distX_offset,
    float max_episode_length,
    torch::Tensor reward_buf
);

torch::Tensor quat_mul(
    torch::Tensor a,
    torch::Tensor b
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("calculate_cartpole_reward", &calculate_cartpole_reward, "CUDA kernel for computing Cartpole reward");
	m.def("calculate_frankacabinet_reward", &calculate_frankacabinet_reward, "CUDA kernel for computing Franka Cabinet reward");
    m.def("calculate_franka_reward_parallel", &calculate_franka_reward_parallel, "CUDA kernel for parallel Franka Cabinet reward");
    m.def("calculate_franka_reward_parallel_stream", &calculate_franka_reward_parallel_stream, "CUDA kernel for parallel Franka Cabinet reward");
    m.def("quat_mul", &quat_mul, "Quaternion multiplication");
}
