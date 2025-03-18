import Quaternions as Q
import Animation as A
import BVH as BVH
import numpy as np

'''relocate anim so that it starts from start_pos'''
def relocate(anim, start_pos):
    delta_pos = start_pos - anim.positions[0, 0]
    anim.positions[:,0] += delta_pos
    return anim

'''rotate anim, so that it starts in different facing direction, delta_q is quaternion'''
def rotate_root(anim, delta_q):
    anim.rotations[:, 0] = delta_q * anim.rotations[:, 0]
    transform = np.repeat(delta_q.transforms(), anim.shape[0], axis=0)
    for f in range(0, anim.shape[0]):
        anim.positions[f,0] = np.matmul(transform[f], anim.positions[f,0])
    return anim

'''naively put anim2 at the end of anim1, not smooth, with teleportation issue'''
def concatenate_naive(anim1, anim2):
    anim1.rotations = Q.Quaternions(np.vstack((anim1.rotations.qs, anim2.rotations.qs)))
    anim1.positions = np.vstack((anim1.positions, anim2.positions))
    return anim1


'''NOTE 
for motion editing, only change anim.rotations and anim.positions
no need to change the skeleton, which is anim.orients and anim.offsets
'''

#TODO: smoothly connects the two motions
def concatenate(anim1, anim2, blend_frame=30):

    rotate1 = anim1.rotations[-blend_frame, 0]
    rotate2 = anim2.rotations[0, 0]
    inverse = Q.Quaternions(-rotate2.qs / np.sum(rotate2.qs**2, axis=-1))
    delta_rot = rotate1 * inverse

    for i in range(anim2.shape[0]):
        anim2.rotations[i, 0] = delta_rot * anim2.rotations[i, 0]
        matrix = delta_rot.transforms()[0]
        anim2.positions[i, 0] = np.dot(matrix, anim2.positions[i, 0])
    delta_pos = anim1.positions[-blend_frame, 0] - anim2.positions[0, 0]
    anim2.positions[:, 0] += delta_pos

    blend_position = np.zeros((blend_frame, anim1.positions.shape[1], 3))
    blend_rotation = np.zeros((blend_frame, anim1.rotations.shape[1], 4))
    for i in range(blend_frame):
        t = i / (blend_frame - 1)
        # (1-t)*p_1 + t*p_2
        blend_position[i] = anim1.positions[-blend_frame + i] * (1 - t) + anim2.positions[i] * t
        blend_rotation[i] = Q.Quaternions.slerp(anim1.rotations[-blend_frame + i], anim2.rotations[i], t).qs

    rotation1 = anim1.rotations.qs[:-blend_frame]
    rotation2 = blend_rotation
    rotation3 = anim2.rotations.qs[blend_frame:]
    new_rotation = np.vstack((rotation1, rotation2, rotation3))
    new_rotation = Q.Quaternions(new_rotation)

    position1 = anim1.positions[:-blend_frame]
    position2 = blend_position
    position3 = anim2.positions[blend_frame:]
    new_position = np.vstack((position1, position2, position3))

    new_anim = A.Animation(new_rotation, new_position, anim1.orients, anim1.offsets, anim1.parents)
    return new_anim


#TODO: splice arm dancing from anim2 into anim1 running
def splice(anim1, anim2, joint_name):

    frame1 = anim1.shape[0]  # anim1 140 frames
    frame2 = anim2.shape[0]  # anim2 510 frames
    joints = anim1.shape[1]
    new_rotation = np.zeros((frame2, joints, 4))
    new_position = np.zeros((frame2, joints, 3))

    for i in range(frame2):
        time = i * (frame1 - 1) / float(frame2 - 1)

        # in [0, frame-1]
        new_frame1 = max(0, min(int(np.floor(time)), frame1 - 1))
        new_frame2 = max(0, min(new_frame1 + 1, frame1 - 1))
        t = time - new_frame1

        position1 = anim1.positions[new_frame1]
        position2 = anim1.positions[new_frame2]
        new_position[i] = (1 - t) * position1 + t * position2

        rotation1 = Q.Quaternions(anim1.rotations.qs[new_frame1])
        rotation2 = Q.Quaternions(anim1.rotations.qs[new_frame2])
        new_rotation[i] = Q.Quaternions.slerp(rotation1, rotation2, t).qs

    new_anim = A.Animation(Q.Quaternions(new_rotation), new_position, anim1.orients, anim1.offsets, anim1.parents)

    # change arms
    arms = ["LeftArm", "RightArm", "LeftHand", "RightForeArm", "LeftForeArm", "LeftFingerBase", "LeftHandIndex1", "RightFingerBase", "RightHand", "RightHandIndex1", "RThumb", "LThumb"]

    arm_parts = []
    for i in arms:
        if i in joint_name:
            arm_parts.append(joint_name.index(i))

    spliced_anim = new_anim.copy()
    for i in range(frame2):
        for j in arm_parts:
            spliced_anim.rotations[i, j] = anim2.rotations[i, j]
            spliced_anim.positions[i, j] = anim2.positions[i, j]
    return spliced_anim


'''load the running motion'''
filepath = ''
filename_run = filepath + '102_08.bvh'
anim_run, joint_names_run, frametime_run = BVH.load(filename_run)

'''load the dance motion'''
filename_dance = filepath + '18_15.bvh'
anim_dance, joint_names_dance, frametime_dance = BVH.load(filename_dance)

# '''simple editing'''
# anim_concat_naive = concatenate_naive(anim_run, anim_dance)
# filename_concat_naive = filepath + 'concat_naive.bvh'
# BVH.save(filename_concat_naive, anim_concat_naive, joint_names_run, frametime_run)

# anim_rotated = rotate_root(anim_run, Q.Quaternions.from_euler(np.array([0, 90, 0])))
# filename_rotated = filepath + 'rotated.bvh'
# BVH.save(filename_rotated, anim_rotated, joint_names_run, frametime_run)

print(joint_names_dance)

result_concatenated = concatenate(anim_run.copy(), anim_dance.copy(), blend_frame=30)
filename_concatenated = filepath + 'result_concatenated.bvh'
BVH.save(filename_concatenated, result_concatenated, joint_names_run, frametime_run)

result_spliced = splice(anim_run.copy(), anim_dance.copy(), joint_names_run)
filename_spliced = filepath + 'result_spliced.bvh'
BVH.save(filename_spliced, result_spliced, joint_names_run, frametime_run)

print('DONE!')
