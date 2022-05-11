#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdio.h>
using namespace torch::indexing;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
	TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
	CHECK_CUDA(x);     \
	CHECK_CONTIGUOUS(x)

template <typename scalar_t> __global__ void cartpole_reward_kernel(scalar_t* pole_angle,
                                                    scalar_t* pole_vel,
                                                    scalar_t* cart_vel,
                                                    scalar_t* cart_pos,
                                                    float reset_dist,
                                                    scalar_t* reset_buf,
                                                    scalar_t* progress_buf,
                                                    float max_episode_length,
                                                    scalar_t* reward_buf,
                                                    int size){
    int num_iters_per_kernel = 1;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid *= num_iters_per_kernel;
	if (tid >= size) return;

    for (int i = 0; i < num_iters_per_kernel; i++){
        float reward = 1.0 - pole_angle[tid + i] * pole_angle[tid + i] - 0.01 * abs(cart_vel[tid + i]) - 0.005 * abs(pole_vel[tid + i]);

        if (abs(cart_pos[tid + i]) > reset_dist || abs(pole_angle[tid + i]) > M_PI_2) {
            reward = -2.0;
        }
        reward_buf[tid + i] = reward;

        if (abs(cart_pos[tid + i]) > reset_dist || abs(pole_angle[tid + i]) > M_PI_2 || progress_buf[tid + i] >= max_episode_length - 1) {
            reset_buf[tid + i] = 1.;
        }
    }
}

void calculate_cartpole_reward(
    torch::Tensor pole_angle,
    torch::Tensor pole_vel,
	torch::Tensor cart_vel,
    torch::Tensor cart_pos,
    float reset_dist,
    torch::Tensor reset_buf,
	torch::Tensor progress_buf,
	float max_episode_length,
    torch::Tensor reward_buf) {
    const int dim_size = (int) pole_angle.size(0);	
    const int threads = 1024;
    const dim3 blocks((dim_size + threads - 1) / threads);

    pole_angle = pole_angle.contiguous();
    pole_vel = pole_vel.contiguous();
    cart_vel = cart_vel.contiguous();
    cart_pos = cart_pos.contiguous();
    reset_buf = reset_buf.contiguous();
    progress_buf = progress_buf.contiguous();
    reward_buf = reward_buf.contiguous();
    
    // std::cout << "START FN DATA" << std::endl;
    // std::cout << pole_angle.index({Slice(0, 5, None)}) << std::endl;
    // std::cout << pole_angle.data<scalar_t>()[0] << std::endl;
    // std::cout << "END FN DATA" << std::endl;

    AT_DISPATCH_FLOATING_TYPES(pole_angle.scalar_type(), "cartpole_reward_forward_cuda", ([&] {
        cartpole_reward_kernel<scalar_t><<<blocks, threads>>>(
        pole_angle.data_ptr<scalar_t>(),
        pole_vel.data_ptr<scalar_t>(),
        cart_vel.data_ptr<scalar_t>(),
        cart_pos.data_ptr<scalar_t>(),
        reset_dist,
        reset_buf.data_ptr<scalar_t>(),
        progress_buf.data_ptr<scalar_t>(),
        max_episode_length,
        reward_buf.data_ptr<scalar_t>(),
        dim_size);
    }));
}

template <typename scalar_t> scalar_t __device__ dot(scalar_t* a, scalar_t* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t> void __device__ cross(scalar_t* a, scalar_t* b, scalar_t* ret) {
    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename scalar_t> void __device__ tf_vector(
    scalar_t* rot,
    scalar_t* axis,
    scalar_t* ret) {
    // shape = b.shape
    // a = a.reshape(-1, 4)
    // b = b.reshape(-1, 3)
    // xyz = a[:, :3]
    // t = xyz.cross(b, dim=-1) * 2
    // return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

    scalar_t xyz[] = {rot[0], rot[1], rot[2]};
    scalar_t t1[3];
    cross<scalar_t>(xyz, axis, t1);

    t1[0] *= 2;
    t1[1] *= 2;
    t1[2] *= 2;

    scalar_t t2[3];
    cross<scalar_t>(xyz, t1, t2);

    ret[0] = axis[0] + rot[3] * t1[0] + t2[0];
    ret[1] = axis[1] + rot[3] * t1[1] + t2[1];
    ret[2] = axis[2] + rot[3] * t1[2] + t2[2];
}

template <typename scalar_t> __global__ void frankacabinet_reward_kernel(
        scalar_t* reset_buf,  // [N]
        scalar_t* progress_buf,  // [N]
        scalar_t* actions,  // [N, 9]
        scalar_t* cabinet_dof_pos,  // [N, 4]
        scalar_t* franka_grasp_pos,  // [N, 3]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_grasp_rot,  // [N, 4]
        scalar_t* drawer_grasp_rot,  // [N, 4]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        scalar_t* gripper_forward_axis,  // [N, 3]
        scalar_t* drawer_inward_axis,  // [N, 3]
        scalar_t* gripper_up_axis,  // [N, 3]
        scalar_t* drawer_up_axis,  // [N, 3]
        int num_envs,
        float dist_reward_scale,
        float rot_reward_scale,
        float around_handle_reward_scale,
        float open_reward_scale,
        float finger_dist_reward_scale,
        float action_penalty_scale,
        float distX_offset,
        float max_episode_length,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* actions_loc = actions + tid * 9;
    scalar_t* cabinet_dof_pos_loc = cabinet_dof_pos + tid * 4;
    scalar_t* franka_grasp_pos_loc = franka_grasp_pos + tid * 3;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_grasp_rot_loc = franka_grasp_rot + tid * 4;
    scalar_t* drawer_grasp_rot_loc = drawer_grasp_rot + tid * 4;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;
    scalar_t* gripper_forward_axis_loc = gripper_forward_axis + tid * 3;
    scalar_t* drawer_inward_axis_loc = drawer_inward_axis + tid * 3;
    scalar_t* gripper_up_axis_loc = gripper_up_axis + tid * 3;
    scalar_t* drawer_up_axis_loc = drawer_up_axis + tid * 3;

    // distance from hand to the drawer
    scalar_t d = pow(franka_grasp_pos_loc[0] - drawer_grasp_pos_loc[0], 2) +
        pow(franka_grasp_pos_loc[1] - drawer_grasp_pos_loc[1], 2) +
        pow(franka_grasp_pos_loc[2] - drawer_grasp_pos_loc[2], 2);
    d = sqrt(d);
    scalar_t dist_reward = 1.0 / (1.0 + pow(d, 2));
    dist_reward *= dist_reward;
    if (d <= 0.02) {
        dist_reward *= 2;
    }

    scalar_t axis1[3];
    tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_forward_axis_loc, axis1);
    scalar_t axis2[3];
    tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_inward_axis_loc, axis2);
    scalar_t axis3[3];
    tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_up_axis_loc, axis3);
    scalar_t axis4[3];
    tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_up_axis_loc, axis4);

    scalar_t dot1 = dot<scalar_t>(axis1, axis2);
    scalar_t dot2 = dot<scalar_t>(axis3, axis4);
    
    // reward for matching the orientation of the hand to the drawer (fingers wrapped)
    scalar_t rot_reward = 0.5 * (copysign(pow(dot1, 2), dot1) + copysign(pow(dot2, 2), dot2));

    // bonus if left finger is above the drawer handle and right below
    scalar_t around_handle_reward = 0;
    if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
        around_handle_reward = 0.5;
    }

    // reward for distance of each finger from the drawer
    scalar_t finger_dist_reward = 0;
    scalar_t lfinger_dist = abs(franka_lfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
    scalar_t rfinger_dist = abs(franka_rfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
    if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
        finger_dist_reward = (0.04 - lfinger_dist) + (0.04 - rfinger_dist);
    }

    // regularization on the actions (summed for each environment)
    scalar_t action_penalty = pow(actions_loc[0], 2) + pow(actions_loc[1], 2) + pow(actions_loc[2], 2) + pow(actions_loc[3], 2) + pow(actions_loc[4], 2) + pow(actions_loc[5], 2) + pow(actions_loc[6], 2) + pow(actions_loc[7], 2) + pow(actions_loc[8], 2);

    // how far the cabinet has been opened out
    scalar_t open_reward = cabinet_dof_pos_loc[3] * around_handle_reward + cabinet_dof_pos_loc[3];

    scalar_t rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward
        + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty;

    // bonus for opening drawer properly
    if (cabinet_dof_pos_loc[3] > 0.01) rewards += 0.5;
    if (cabinet_dof_pos_loc[3] > 0.2) rewards += around_handle_reward;
    if (cabinet_dof_pos_loc[3] > 0.39) rewards += 2 * around_handle_reward;

    // prevent bad style in opening drawer
    if (franka_lfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) rewards = -1;
    if (franka_rfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) rewards = -1;

    // reset if drawer is open or max length reached
    if (cabinet_dof_pos_loc[3] > 0.39) reset_buf[tid] = 1;
    if (progress_buf[tid] >= max_episode_length - 1) reset_buf[tid] = 1;

    reward_buf[tid] = rewards;
}

template <typename scalar_t> __global__ void franka_parallel_kernel1(
        scalar_t* reset_buf,  // [N]
        scalar_t* progress_buf,  // [N]
        scalar_t* actions,  // [N, 9]
        scalar_t* cabinet_dof_pos,  // [N, 4]
        scalar_t* franka_grasp_pos,  // [N, 3]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_grasp_rot,  // [N, 4]
        scalar_t* drawer_grasp_rot,  // [N, 4]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        scalar_t* gripper_forward_axis,  // [N, 3]
        scalar_t* drawer_inward_axis,  // [N, 3]
        scalar_t* gripper_up_axis,  // [N, 3]
        scalar_t* drawer_up_axis,  // [N, 3]
        int num_envs,
        float dist_reward_scale,
        float rot_reward_scale,
        float around_handle_reward_scale,
        float open_reward_scale,
        float finger_dist_reward_scale,
        float action_penalty_scale,
        float distX_offset,
        float max_episode_length,
        scalar_t* reward_buf  // [N]
        ){
	int real_tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (real_tid >= num_envs*4) return;
    int tid = real_tid % num_envs;

    scalar_t* actions_loc = actions + tid * 9;
    scalar_t* cabinet_dof_pos_loc = cabinet_dof_pos + tid * 4;
    scalar_t* franka_grasp_pos_loc = franka_grasp_pos + tid * 3;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_grasp_rot_loc = franka_grasp_rot + tid * 4;
    scalar_t* drawer_grasp_rot_loc = drawer_grasp_rot + tid * 4;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;
    scalar_t* gripper_forward_axis_loc = gripper_forward_axis + tid * 3;
    scalar_t* drawer_inward_axis_loc = drawer_inward_axis + tid * 3;
    scalar_t* gripper_up_axis_loc = gripper_up_axis + tid * 3;
    scalar_t* drawer_up_axis_loc = drawer_up_axis + tid * 3;

    // distance from hand to the drawer
    if (real_tid / num_envs == 0) {
        scalar_t d = pow(franka_grasp_pos_loc[0] - drawer_grasp_pos_loc[0], 2) +
            pow(franka_grasp_pos_loc[1] - drawer_grasp_pos_loc[1], 2) +
            pow(franka_grasp_pos_loc[2] - drawer_grasp_pos_loc[2], 2);
        d = sqrt(d);
        scalar_t dist_reward = 1.0 / (1.0 + pow(d, 2));
        dist_reward *= dist_reward;
        if (d <= 0.02) {
            dist_reward *= 2;
        }
        atomicAdd(reward_buf + tid, dist_reward_scale * dist_reward);
    }

    if (real_tid / num_envs == 1) {
        scalar_t axis1[3];
        tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_forward_axis_loc, axis1);
        scalar_t axis2[3];
        tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_inward_axis_loc, axis2);
        scalar_t axis3[3];
        tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_up_axis_loc, axis3);
        scalar_t axis4[3];
        tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_up_axis_loc, axis4);

        scalar_t dot1 = dot<scalar_t>(axis1, axis2);
        scalar_t dot2 = dot<scalar_t>(axis3, axis4);
        
        // reward for matching the orientation of the hand to the drawer (fingers wrapped)
        scalar_t rot_reward = 0.5 * (copysign(pow(dot1, 2), dot1) + copysign(pow(dot2, 2), dot2));
        atomicAdd(reward_buf + tid, rot_reward_scale * rot_reward);
    }
    
    if (real_tid / num_envs == 2) {
        // bonus if left finger is above the drawer handle and right below
        scalar_t around_handle_reward = 0;
        if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
            around_handle_reward = 0.5;
        }
        // how far the cabinet has been opened out
        scalar_t open_reward = cabinet_dof_pos_loc[3] * around_handle_reward + cabinet_dof_pos_loc[3];
        scalar_t rewards = around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward;
        // bonus for opening drawer properly
        if (cabinet_dof_pos_loc[3] > 0.01) rewards += 0.5;
        if (cabinet_dof_pos_loc[3] > 0.2) rewards += around_handle_reward;
        if (cabinet_dof_pos_loc[3] > 0.39) rewards += 2 * around_handle_reward;
        atomicAdd(reward_buf + tid, rewards);
    }
    
    if (real_tid / num_envs == 3) {
        // reward for distance of each finger from the drawer
        scalar_t finger_dist_reward = 0;
        scalar_t lfinger_dist = abs(franka_lfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
        scalar_t rfinger_dist = abs(franka_rfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
        if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
            finger_dist_reward = (0.04 - lfinger_dist) + (0.04 - rfinger_dist);
        }
        // regularization on the actions (summed for each environment)
        scalar_t action_penalty = pow(actions_loc[0], 2) + pow(actions_loc[1], 2) + pow(actions_loc[2], 2) + pow(actions_loc[3], 2) + pow(actions_loc[4], 2) + pow(actions_loc[5], 2) + pow(actions_loc[6], 2) + pow(actions_loc[7], 2) + pow(actions_loc[8], 2);
        atomicAdd(reward_buf + tid, finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty);
    }
}

template <typename scalar_t> __global__ void franka_parallel_kernel2(
        scalar_t* reset_buf,  // [N]
        scalar_t* progress_buf,  // [N]
        scalar_t* cabinet_dof_pos,  // [N, 4]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        int num_envs,
        float distX_offset,
        float max_episode_length,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* cabinet_dof_pos_loc = cabinet_dof_pos + tid * 4;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;

    // prevent bad style in opening drawer
    if (franka_lfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) reward_buf[tid] = -1;
    if (franka_rfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) reward_buf[tid] = -1;

    // reset if drawer is open or max length reached
    if (cabinet_dof_pos_loc[3] > 0.39) reset_buf[tid] = 1;
    if (progress_buf[tid] >= max_episode_length - 1) reset_buf[tid] = 1;
}

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
) {
    const int threads = 128;
    const dim3 blocks((num_envs + threads - 1) / threads);

    reset_buf = reset_buf.contiguous();
    progress_buf = progress_buf.contiguous();
    actions = actions.contiguous();
    cabinet_dof_pos = cabinet_dof_pos.contiguous();
    franka_grasp_pos = franka_grasp_pos.contiguous();
    drawer_grasp_pos = drawer_grasp_pos.contiguous();
    franka_grasp_rot = franka_grasp_rot.contiguous();
    drawer_grasp_rot = drawer_grasp_rot.contiguous();
    franka_lfinger_pos = franka_lfinger_pos.contiguous();
    franka_rfinger_pos = franka_rfinger_pos.contiguous();
    gripper_forward_axis = gripper_forward_axis.contiguous();
    drawer_inward_axis = drawer_inward_axis.contiguous();
    gripper_up_axis = gripper_up_axis.contiguous();
    drawer_up_axis = drawer_up_axis.contiguous();
    reward_buf = reward_buf.contiguous();

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "frankacabinet_reward_forward_cuda", ([&] {
        frankacabinet_reward_kernel<scalar_t><<<blocks, threads>>>(
        reset_buf.data_ptr<scalar_t>(),
        progress_buf.data_ptr<scalar_t>(),
        actions.data_ptr<scalar_t>(),
        cabinet_dof_pos.data_ptr<scalar_t>(),
        franka_grasp_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_grasp_rot.data_ptr<scalar_t>(),
        drawer_grasp_rot.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        gripper_forward_axis.data_ptr<scalar_t>(),
        drawer_inward_axis.data_ptr<scalar_t>(),
        gripper_up_axis.data_ptr<scalar_t>(),
        drawer_up_axis.data_ptr<scalar_t>(),
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        reward_buf.data_ptr<scalar_t>());
    }));
}

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
) {
    const int threads = 128;
    const dim3 blocks((num_envs*4 + threads - 1) / threads);
    const dim3 blocks2((num_envs + threads - 1) / threads);

    reset_buf = reset_buf.contiguous();
    progress_buf = progress_buf.contiguous();
    actions = actions.contiguous();
    cabinet_dof_pos = cabinet_dof_pos.contiguous();
    franka_grasp_pos = franka_grasp_pos.contiguous();
    drawer_grasp_pos = drawer_grasp_pos.contiguous();
    franka_grasp_rot = franka_grasp_rot.contiguous();
    drawer_grasp_rot = drawer_grasp_rot.contiguous();
    franka_lfinger_pos = franka_lfinger_pos.contiguous();
    franka_rfinger_pos = franka_rfinger_pos.contiguous();
    gripper_forward_axis = gripper_forward_axis.contiguous();
    drawer_inward_axis = drawer_inward_axis.contiguous();
    gripper_up_axis = gripper_up_axis.contiguous();
    drawer_up_axis = drawer_up_axis.contiguous();
    reward_buf = reward_buf.contiguous();
    reward_buf.zero_();


    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_parallel_kernel1", ([&] {
        franka_parallel_kernel1<scalar_t><<<blocks, threads>>>(
        reset_buf.data_ptr<scalar_t>(),
        progress_buf.data_ptr<scalar_t>(),
        actions.data_ptr<scalar_t>(),
        cabinet_dof_pos.data_ptr<scalar_t>(),
        franka_grasp_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_grasp_rot.data_ptr<scalar_t>(),
        drawer_grasp_rot.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        gripper_forward_axis.data_ptr<scalar_t>(),
        drawer_inward_axis.data_ptr<scalar_t>(),
        gripper_up_axis.data_ptr<scalar_t>(),
        drawer_up_axis.data_ptr<scalar_t>(),
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        reward_buf.data_ptr<scalar_t>());
    }));

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_parallel_kernel2", ([&] {
        franka_parallel_kernel2<scalar_t><<<blocks2, threads>>>(
        reset_buf.data_ptr<scalar_t>(),
        progress_buf.data_ptr<scalar_t>(),
        cabinet_dof_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        num_envs,
        distX_offset,
        max_episode_length,
        reward_buf.data_ptr<scalar_t>());
    }));
}

template <typename scalar_t> __global__ void franka_reward_kernel1(
        scalar_t* franka_grasp_pos,  // [N, 3]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        int num_envs,
        float dist_reward_scale,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* franka_grasp_pos_loc = franka_grasp_pos + tid * 3;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;

    // distance from hand to the drawer
    scalar_t d = pow(franka_grasp_pos_loc[0] - drawer_grasp_pos_loc[0], 2) +
        pow(franka_grasp_pos_loc[1] - drawer_grasp_pos_loc[1], 2) +
        pow(franka_grasp_pos_loc[2] - drawer_grasp_pos_loc[2], 2);
    d = sqrt(d);
    scalar_t dist_reward = 1.0 / (1.0 + pow(d, 2));
    dist_reward *= dist_reward;
    if (d <= 0.02) {
        dist_reward *= 2;
    }
    atomicAdd(reward_buf + tid, dist_reward_scale * dist_reward);
}

template <typename scalar_t> __global__ void franka_reward_kernel2(
        scalar_t* franka_grasp_rot,  // [N, 4]
        scalar_t* drawer_grasp_rot,  // [N, 4]
        scalar_t* gripper_forward_axis,  // [N, 3]
        scalar_t* drawer_inward_axis,  // [N, 3]
        scalar_t* gripper_up_axis,  // [N, 3]
        scalar_t* drawer_up_axis,  // [N, 3]
        int num_envs,
        float rot_reward_scale,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* franka_grasp_rot_loc = franka_grasp_rot + tid * 4;
    scalar_t* drawer_grasp_rot_loc = drawer_grasp_rot + tid * 4;
    scalar_t* gripper_forward_axis_loc = gripper_forward_axis + tid * 3;
    scalar_t* drawer_inward_axis_loc = drawer_inward_axis + tid * 3;
    scalar_t* gripper_up_axis_loc = gripper_up_axis + tid * 3;
    scalar_t* drawer_up_axis_loc = drawer_up_axis + tid * 3;

    scalar_t axis1[3];
    tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_forward_axis_loc, axis1);
    scalar_t axis2[3];
    tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_inward_axis_loc, axis2);
    scalar_t axis3[3];
    tf_vector<scalar_t>(franka_grasp_rot_loc, gripper_up_axis_loc, axis3);
    scalar_t axis4[3];
    tf_vector<scalar_t>(drawer_grasp_rot_loc, drawer_up_axis_loc, axis4);

    scalar_t dot1 = dot<scalar_t>(axis1, axis2);
    scalar_t dot2 = dot<scalar_t>(axis3, axis4);
    
    // reward for matching the orientation of the hand to the drawer (fingers wrapped)
    scalar_t rot_reward = 0.5 * (copysign(pow(dot1, 2), dot1) + copysign(pow(dot2, 2), dot2));
    atomicAdd(reward_buf + tid, rot_reward_scale * rot_reward);
}

template <typename scalar_t> __global__ void franka_reward_kernel3(
        scalar_t* cabinet_dof_pos,  // [N, 4]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        int num_envs,
        float around_handle_reward_scale,
        float open_reward_scale,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* cabinet_dof_pos_loc = cabinet_dof_pos + tid * 4;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;

    // bonus if left finger is above the drawer handle and right below
    scalar_t around_handle_reward = 0;
    if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
        around_handle_reward = 0.5;
    }
    // how far the cabinet has been opened out
    scalar_t open_reward = cabinet_dof_pos_loc[3] * around_handle_reward + cabinet_dof_pos_loc[3];
    scalar_t rewards = around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward;
    // bonus for opening drawer properly
    if (cabinet_dof_pos_loc[3] > 0.01) rewards += 0.5;
    if (cabinet_dof_pos_loc[3] > 0.2) rewards += around_handle_reward;
    if (cabinet_dof_pos_loc[3] > 0.39) rewards += 2 * around_handle_reward;
    atomicAdd(reward_buf + tid, rewards);
}

template <typename scalar_t> __global__ void franka_reward_kernel4(
        scalar_t* actions,  // [N, 9]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        int num_envs,
        float finger_dist_reward_scale,
        float action_penalty_scale,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* actions_loc = actions + tid * 9;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;

    // reward for distance of each finger from the drawer
    scalar_t finger_dist_reward = 0;
    scalar_t lfinger_dist = abs(franka_lfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
    scalar_t rfinger_dist = abs(franka_rfinger_pos_loc[2] - drawer_grasp_pos_loc[2]);
    if (franka_lfinger_pos_loc[2] > drawer_grasp_pos_loc[2] && franka_rfinger_pos_loc[2] < drawer_grasp_pos_loc[2]) {
        finger_dist_reward = (0.04 - lfinger_dist) + (0.04 - rfinger_dist);
    }
    // regularization on the actions (summed for each environment)
    scalar_t action_penalty = pow(actions_loc[0], 2) + pow(actions_loc[1], 2) + pow(actions_loc[2], 2) + pow(actions_loc[3], 2) + pow(actions_loc[4], 2) + pow(actions_loc[5], 2) + pow(actions_loc[6], 2) + pow(actions_loc[7], 2) + pow(actions_loc[8], 2);
    atomicAdd(reward_buf + tid, finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty);
}

template <typename scalar_t> __global__ void franka_reward_kernel5(
        scalar_t* reset_buf,  // [N]
        scalar_t* progress_buf,  // [N]
        scalar_t* cabinet_dof_pos,  // [N, 4]
        scalar_t* drawer_grasp_pos,  // [N, 3]
        scalar_t* franka_lfinger_pos,  // [N, 3]
        scalar_t* franka_rfinger_pos,  // [N, 3]
        int num_envs,
        float distX_offset,
        float max_episode_length,
        scalar_t* reward_buf  // [N]
        ){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t* cabinet_dof_pos_loc = cabinet_dof_pos + tid * 4;
    scalar_t* drawer_grasp_pos_loc = drawer_grasp_pos + tid * 3;
    scalar_t* franka_lfinger_pos_loc = franka_lfinger_pos + tid * 3;
    scalar_t* franka_rfinger_pos_loc = franka_rfinger_pos + tid * 3;

    // prevent bad style in opening drawer
    if (franka_lfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) reward_buf[tid] = -1;
    if (franka_rfinger_pos_loc[0] < drawer_grasp_pos_loc[0] - distX_offset) reward_buf[tid] = -1;

    // reset if drawer is open or max length reached
    if (cabinet_dof_pos_loc[3] > 0.39) reset_buf[tid] = 1;
    if (progress_buf[tid] >= max_episode_length - 1) reset_buf[tid] = 1;
}

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
) {
    const int threads = 128;
    const dim3 blocks((num_envs + threads - 1) / threads);

    reset_buf = reset_buf.contiguous();
    progress_buf = progress_buf.contiguous();
    actions = actions.contiguous();
    cabinet_dof_pos = cabinet_dof_pos.contiguous();
    franka_grasp_pos = franka_grasp_pos.contiguous();
    drawer_grasp_pos = drawer_grasp_pos.contiguous();
    franka_grasp_rot = franka_grasp_rot.contiguous();
    drawer_grasp_rot = drawer_grasp_rot.contiguous();
    franka_lfinger_pos = franka_lfinger_pos.contiguous();
    franka_rfinger_pos = franka_rfinger_pos.contiguous();
    gripper_forward_axis = gripper_forward_axis.contiguous();
    drawer_inward_axis = drawer_inward_axis.contiguous();
    gripper_up_axis = gripper_up_axis.contiguous();
    drawer_up_axis = drawer_up_axis.contiguous();
    reward_buf = reward_buf.contiguous();
    reward_buf.zero_();

    cudaStream_t stream[4];
    for (int i = 0; i < 4; ++i)
        cudaStreamCreate(&stream[i]);

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_kernel1", ([&] {
        franka_reward_kernel1<scalar_t><<<blocks, threads, 0, stream[0]>>>(
        franka_grasp_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        num_envs,
        dist_reward_scale,
        reward_buf.data_ptr<scalar_t>());
    }));

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_kernel2", ([&] {
        franka_reward_kernel2<scalar_t><<<blocks, threads, 0, stream[1]>>>(
        franka_grasp_rot.data_ptr<scalar_t>(),
        drawer_grasp_rot.data_ptr<scalar_t>(),
        gripper_forward_axis.data_ptr<scalar_t>(),
        drawer_inward_axis.data_ptr<scalar_t>(),
        gripper_up_axis.data_ptr<scalar_t>(),
        drawer_up_axis.data_ptr<scalar_t>(),
        num_envs,
        rot_reward_scale,
        reward_buf.data_ptr<scalar_t>());
    }));

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_kernel3", ([&] {
        franka_reward_kernel3<scalar_t><<<blocks, threads, 0, stream[2]>>>(
        cabinet_dof_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        num_envs,
        around_handle_reward_scale,
        open_reward_scale,
        reward_buf.data_ptr<scalar_t>());
    }));

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_kernel4", ([&] {
        franka_reward_kernel4<scalar_t><<<blocks, threads, 0, stream[3]>>>(
        actions.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        num_envs,
        finger_dist_reward_scale,
        action_penalty_scale,
        reward_buf.data_ptr<scalar_t>());
    }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(actions.scalar_type(), "franka_kernel5", ([&] {
        franka_reward_kernel5<scalar_t><<<blocks, threads>>>(
        reset_buf.data_ptr<scalar_t>(),
        progress_buf.data_ptr<scalar_t>(),
        cabinet_dof_pos.data_ptr<scalar_t>(),
        drawer_grasp_pos.data_ptr<scalar_t>(),
        franka_lfinger_pos.data_ptr<scalar_t>(),
        franka_rfinger_pos.data_ptr<scalar_t>(),
        num_envs,
        distX_offset,
        max_episode_length,
        reward_buf.data_ptr<scalar_t>());
    }));

    cudaDeviceSynchronize();

    for (int i = 0; i < 4; ++i)
        cudaStreamDestroy(stream[i]);
}

template <typename scalar_t> __global__ void quat_mul_kernel(
    scalar_t* q1,
    scalar_t* q2,
    scalar_t* ret,
    int num_envs
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_envs) return;

    scalar_t x1 = q1[tid*4 + 0];
    scalar_t y1 = q1[tid*4 + 1];
    scalar_t z1 = q1[tid*4 + 2];
    scalar_t w1 = q1[tid*4 + 3];
    scalar_t x2 = q2[tid*4 + 0];
    scalar_t y2 = q2[tid*4 + 1];
    scalar_t z2 = q2[tid*4 + 2];
    scalar_t w2 = q2[tid*4 + 3];

    scalar_t ww = (z1 + x1) * (x2 + y2);
    scalar_t yy = (w1 - y1) * (w2 + z2);
    scalar_t zz = (w1 + y1) * (w2 - z2);
    scalar_t xx = ww + yy + zz;
    scalar_t qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));
    scalar_t w = qq - ww + (z1 - y1) * (y2 - z2);
    scalar_t x = qq - xx + (x1 + w1) * (x2 + w2);
    scalar_t y = qq - yy + (w1 - x1) * (y2 + z2);
    scalar_t z = qq - zz + (z1 + y1) * (w2 - x2);

    ret[tid*4 + 0] = x;
    ret[tid*4 + 1] = y;
    ret[tid*4 + 2] = z;
    ret[tid*4 + 3] = w;
}

torch::Tensor quat_mul(
    torch::Tensor a,
    torch::Tensor b)
{
    // torch::Tensor a_reshaped = a.permute({1, 0})//.contiguous();
    // torch::Tensor b_reshaped = b.permute({1, 0})//.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    torch::Tensor ret = torch::clone(a); //.contiguous();

    const int dim_size = (int) a.size(0);

    const int threads = 32;
    const dim3 blocks((dim_size + threads - 1) / threads);

    // auto x1 = a_reshaped.index({0});
    // auto y1 = a_reshaped.index({1});
    // auto z1 = a_reshaped.index({2});
    // auto w1 = a_reshaped.index({3});

    // auto x2 = b_reshaped.index({0});
    // auto y2 = b_reshaped.index({1});
    // auto z2 = b_reshaped.index({2});
    // auto w2 = b_reshaped.index({3});

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "quat_mult", ([&] {
        quat_mul_kernel<scalar_t><<<blocks, threads>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        ret.data_ptr<scalar_t>(),
        dim_size);
    }));

    return ret;
}

