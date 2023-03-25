# some code have been reused from https://github.com/una-dinosauria/3d-pose-baseline
import numpy as np
import os
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset

class Human36M(Dataset):
    
    def load_source_data(self, bpath, dim, dim_to_use, disp=False ):
        """
        Loads 3d/ 2d ground truth from disk, and
        puts it in an easy-to-acess dictionary
        Args
        bpath: String. Path where to load the data from
        subjects: List of integers. Subjects whose data will be loaded
        actions: List of strings. The actions to load
        dim: Integer={2,3}. Load 2 or 3-dimensional data
        dim_to_use: out of 32 joints, select 17 joints
        Returns:
        data: Dictionary with keys k=(subject, action, seqname)
          values v=(nx(17*2) matrix of 2d/3d ground truth)
        """
        if not dim in [2,3]:
            raise(ValueError, 'dim must be 2 or 3')

        data = {}
        total_recs=0
        sub_actions = []

        for subj in self.subjects:
            if dim == 3:
                dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D3_Positions_mono/*.cdf.mat')
            elif dim == 2:
                dpath =  os.path.join( bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D2_Positions/*.cdf.mat')

            fnames = glob.glob( dpath )
            num_recs=0

            for fname in fnames:
                seqname = os.path.basename( fname )

                action = seqname.split(".")[0]
                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                cam = seqname.split(".")[1]
                cam_id = self.h36m_cameras_intrinsic_params[cam]['id']

                if subj == 11 and action == "Directions":
                    continue # corrupt video

                sub_actions.append(action)
                poses = sio.loadmat(fname)['data'][0][0]
                poses = poses.reshape(-1, 32, dim)[:, dim_to_use]
                data[ (subj, action, cam_id) ] = poses
                num_recs+=poses.shape[0]
                                

            if disp:
                print("subject: ", subj, " num_files: ", len(fnames), " num_recs: ", num_recs)
            total_recs+=num_recs

        sub_actions = np.asarray(sub_actions)
        sub_actions = np.unique(sub_actions)
        if disp:
            print("load_source_data / records loaded: ", total_recs, " x ", poses.shape[1], " x ", poses.shape[2], "\n")

        return data, sub_actions


    def load_cpn_detection(self, cpn_file="data_2d_h36m_detectron_ft_h36m.npz", disp=False):
        """
        Loads 2D prediction by CPN from disk
        """
        keypoints = np.load(cpn_file, allow_pickle=True)
        keypoints = keypoints['positions_2d'].item() # a nested dict with Subject->Action

        # Check for >= instead of == because some videos in H3.6M contain extra frames
        
        for subj in self.subjects:
            subject = "S"+str(subj)
            
            for action in keypoints[subject].keys():
                assert action in self.actions, 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)

                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = self.data_2d[(subj,action,cam_idx)].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        data = {}
        total_recs=0
        sub_actions = []

        for subj in self.subjects:
            subject = "S" + str(subj)
            subj_keypoints = keypoints[subject]
            num_recs=0
            num_files = 0

            for action in subj_keypoints.keys():
                
                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                if subject == 'S11' and action == "Directions":
                    continue # corrupt video

                for cam_id, kps in enumerate(subj_keypoints[action]):    
                    sub_actions.append(action)
                    poses = kps # -1, 17, 2
                    poses = np.array(poses)
                    data[ (subj, action, cam_id) ] = poses
                    num_recs+= poses.shape[0]  
                    num_files+=1                  

            if disp:
                print("subject: ", subj, " num_files: ", num_files, "num_recs: ", num_recs)
            total_recs+=num_recs

        sub_actions = np.asarray(sub_actions)
        sub_actions = np.unique(sub_actions)
        if disp:
            print("load_cpn_detection / records loaded: ", total_recs, " x ", poses.shape[1], " x ", poses.shape[2], "\n")

        return data, sub_actions

    def postprocess_3d(self, poses_set, dim_to_use, with_hip=True ):
        """
        Center 3d points around root
        Args
        poses_set: Dictionary with keys k=(subject, action, seqname)
        and value v=(nx(17*3) matrix of 3d ground truth)
        Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
        """
        root_positions = {}
        for k in poses_set.keys():
            # Subtract the root from the 1st position onwards
            # 0th position tracks the root position for future use
            poses = poses_set[k]
            root_joint = poses[:, :1]
            if with_hip:
                # CHANGE [:, 1:] to [:, :] AFTER VISUALIZATION
                poses[:, :] -= root_joint
            else:
                poses[:, 1:] -= root_joint
            poses_set[k] = poses

        return poses_set


    def normalize_2d(self, data, dim_to_use):
        """
        Normalize so that [0, Width] is mapped to [-1, 1], while preserving the aspect ratio
        """
        data_out = {}

        for key in data.keys():
            # extract cam id from filename
            subj, action, cam_id = key

            for k, v in self.h36m_cameras_intrinsic_params.items():
                if v['id'] == cam_id:
                    camera_id = k

            joint_data = data[ key ]

            # get cam resolution
            height = self.h36m_cameras_intrinsic_params[camera_id]['res_h']
            width = self.h36m_cameras_intrinsic_params[camera_id]['res_w']

            # Normalize
            data_out[key] = joint_data/width*2 -[1, height/width]

        return data_out
    

    def postprocess_2d(self, poses_set, with_hip=True ):
        """
        Center 2d points around root
        Args
        poses_set: Dictionary with keys k=(subject, action, seqname)
        and value v=(nx(17*2) matrix of 2d ground truth)
        Returns
        poses_set: dictionary with 2d data centred around root (center hip) joint
        root_positions: dictionary with the original 2d position of each pose
        """
        root_positions = {}
        poses_set_centered = {}
        for k in poses_set.keys():
            # Subtract the root from the 1st position onwards
            # 0th position tracks the root position for future use
            poses_centered = poses_set[k].copy()
            root_joint = poses_centered[:, :1]
            if with_hip:
                poses_centered[:, :] -= root_joint
            else:
                poses_centered[:, 1:] -= root_joint
            poses_set_centered[k] = poses_centered

        return poses_set_centered
    

    def get_h36m_joint_id_to_name(self):
        """
        return h36m id to name mapping in dict format
        """
        joints = np.array(self.H36M_NAMES)[self.dim_to_use]
        # replace neck/nose with neck
        joints[joints=='Neck/Nose']='Neck'
        num_nodes = len(joints)
        joint_id_to_names = {i: joint for i, joint in enumerate(joints)}
        return joint_id_to_names
        

    def get_h36m_joint_name_to_id(self):
        """
        return h36m name to id mapping in dict format
        """
        joints = np.array(self.H36M_NAMES)[self.dim_to_use]
        # replace neck/nose with neck
        joints[joints=='Neck/Nose']='Neck'
        num_nodes = len(joints)
        joint_names_to_id = {joint: i for i, joint in enumerate(joints)}
        return joint_names_to_id

    def get_h36m_joint_id_to_name_ex_hip(self):
        """
        return h36m id to name mapping in dict format excluding hip
        """
        joints = np.array(self.H36M_NAMES)[self.dim_to_use]
        # replace neck/nose with neck
        joints[joints=='Neck/Nose']='Neck'
        num_nodes = len(joints) -1
        joint_id_to_names = {i: joint for i, joint in enumerate(joints[1:])}
        return joint_id_to_names
        

    def get_h36m_joint_name_to_id_ex_hip(self):
        """
        return h36m name to id mapping in dict format excluding hip
        """
        joints = np.array(self.H36M_NAMES)[self.dim_to_use]
        # replace neck/nose with neck
        joints[joints=='Neck/Nose']='Neck'
        num_nodes = len(joints)
        joint_names_to_id = {joint: i for i, joint in enumerate(joints[1:])}
        return joint_names_to_id

    def __init__(self, data_dir, cpn_file, train=True, with_hip=True, ds_category="gt", actions="all"):
        """
        set joint names /dim_to_use
        set training and test subjects
        set camera resolutions
        load 2d (normalized) to input and 3d skeleton (unnormalized, root centered) to output
        """

        # Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
        self.H36M_NAMES = ['']*32
        self.H36M_NAMES[0]  = 'Hip'
        self.H36M_NAMES[1]  = 'RHip'
        self.H36M_NAMES[2]  = 'RKnee'
        self.H36M_NAMES[3]  = 'RFoot'
        self.H36M_NAMES[6]  = 'LHip'
        self.H36M_NAMES[7]  = 'LKnee'
        self.H36M_NAMES[8]  = 'LFoot'
        self.H36M_NAMES[12] = 'Spine'
        self.H36M_NAMES[13] = 'Thorax'
        self.H36M_NAMES[14] = 'Neck/Nose'
        self.H36M_NAMES[15] = 'Head'
        self.H36M_NAMES[17] = 'LShoulder'
        self.H36M_NAMES[18] = 'LElbow'
        self.H36M_NAMES[19] = 'LWrist'
        self.H36M_NAMES[25] = 'RShoulder'
        self.H36M_NAMES[26] = 'RElbow'
        self.H36M_NAMES[27] = 'RWrist'
        
        self.dim_to_use = np.where(np.array([x != '' for x in self.H36M_NAMES]))[0]
        
        if train:
            self.subjects = [1, 5, 6, 7, 8]
        else:
            self.subjects = [9, 11]

        # dataset category gt/cpn/globalpos
        self.ds_category = ds_category
            
        self.h36m_cameras_intrinsic_params = {
            '54138969': {
                'id': 0,
                'res_w': 1000,
                'res_h': 1002
            },
            '55011271': {
                'id': 1,
                'res_w': 1000,
                'res_h': 1000
            },
            '58860488': {
                'id': 2,
                'res_w': 1000,
                'res_h': 1000
            },
            '60457274': {
                'id': 3,
                'res_w': 1000,
                'res_h': 1002
            }
        }
        
        self.input , self.output, self.hips, self.input_globalpos, self.input_cpn = [], [], [], [], []

        # load 2d keypoints (camera frame)
        self.data_2d, self.actions = self.load_source_data(data_dir, 2, self.dim_to_use)
        self.data_2d_cpn, _ = self.load_cpn_detection(cpn_file=cpn_file)

        # print(self.actions)
        
        # load 3d ground truth camera frame positions
        self.data_3d, _ = self.load_source_data(data_dir, 3, self.dim_to_use)
        
        # root center 3d skeleton
        self.data_3d = self.postprocess_3d(self.data_3d, self.dim_to_use, with_hip)

        # normalize 2d skeleton -1 to +1
        self.data_2d = self.normalize_2d(self.data_2d, self.dim_to_use)
        self.data_2d_cpn = self.normalize_2d(self.data_2d_cpn, self.dim_to_use)

        # root center 2d skeleton
        self.data_2d_centered = self.postprocess_2d(self.data_2d, with_hip)
        self.data_2d_cpn_centered = self.postprocess_2d(self.data_2d_cpn, with_hip)

        first_joint_id = 0 if with_hip else 1

        for key in self.data_2d.keys():
            sub, act, cam_id = key
            pose_2d = self.data_2d[key]
            pose_2d_centered = self.data_2d_centered[key] 
            pose_2d_cpn = self.data_2d_cpn[key]         
            pose_2d_cpn_centered = self.data_2d_cpn_centered[key]         
            pose_3d = self.data_3d[key]

            if actions != "all":
                if not (actions in act):
                    continue

            for i in range(pose_2d.shape[0]):
                # self.input.append(pose_2d_centered[i][first_joint_id:])
                self.input.append(pose_2d[i][first_joint_id:])
                
                # self.input_cpn.append(pose_2d_cpn_centered[i][first_joint_id:])
                self.input_cpn.append(pose_2d_cpn[i][first_joint_id:])
                
                self.output.append(pose_3d[i][first_joint_id:])

                ##################### extra - for learning globalpos ######################
                self.hips.append(pose_3d[i][0])
                # global gt
                self.input_globalpos.append(pose_2d[i])
                # global cpn
                # self.input_globalpos.append(pose_2d_cpn[i])


    def __getitem__(self, idx):  
        '''
        Get normalized 2d keypoints and unnormalized 3d skeleton
        :param idx:
        :return:
            2d keypoints:  Tensor    17 * 2
            3d keypoints:  Tensor    17 * 3
            
        '''
        if self.ds_category == "gt":
            inputs = torch.from_numpy(self.input[idx]).float()
            outputs = torch.from_numpy(self.output[idx]).float()
            sample = {'inputs': inputs, 'outputs': outputs}
        elif self.ds_category == "cpn":
            inputs = torch.from_numpy(self.input_cpn[idx]).float()
            inputs_gt = torch.from_numpy(self.input[idx]).float()
            outputs = torch.from_numpy(self.output[idx]).float()
            sample = {'inputs': inputs, 'outputs': outputs, 'inputs_gt': inputs_gt}
        elif self.ds_category == "globalpos":
            hips = torch.from_numpy(self.hips[idx]).float()
            inputs_globalpos = torch.from_numpy(self.input_globalpos[idx]).float()
            sample = {'inputs_globalpos': inputs_globalpos, 'hips': hips }
        return sample
    
    def __len__(self):
        return len(self.input)


# test_dataset = Human36M(data_dir='/media/HDD4/datasets/Human3.6M/pose_zip', cpn_file="/media/HDD4/datasets/Human3.6M/data_2d_h36m_cpn_ft_h36m_dbb.npz", train=False, with_hip=True, ds_category='cpn')
# data = test_dataset.__getitem__(0)
# batch_inputs = data['inputs']
# batch_gt = data['outputs'] 
# print(batch_inputs.shape, batch_gt.shape)
