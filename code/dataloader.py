import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import re
from scipy.spatial.transform import Rotation as R
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from dataprocess.mano_layer_self import MANO_SMPL, Render

# Path to the MANO_RIGHT model pickle file
mano_path = 'D:/OneDrive/code/mano_v1_2/models/MANO_RIGHT.pkl'
mano_layer = MANO_SMPL(mano_path)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to a 3x3 rotation matrix using Gram-Schmidt orthogonalization.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        3x3 rotation matrix, of size (*, 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)  # Normalize the first vector
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1  # Make b2 orthogonal to b1
    b2 = F.normalize(b2, dim=-1)  # Normalize the second vector
    b3 = torch.cross(b1, b2, dim=-1)  # Compute the third vector using cross product
    return torch.stack((b1, b2, b3), dim=-2)  # Stack the vectors to form a rotation matrix


class SelfDataset(Dataset):
    def __init__(self, data_root, label_root, camera_id, kind, cam_paras, cube_size=250,
                 use_imu=False, window_before=10, window_after=10, crop_size=(224, 224)):
        """
        Args:
            data_root (str): Root directory of the dataset containing image and IMU data
            label_root (str): Directory of annotation (joint positions, MANO parameters)
            camera_id (str): Identifier for the camera view (e.g., 'camera0')
            kind (str): Dataset type, 'train', 'val', or 'test'
            cam_paras (torch.Tensor): Camera intrinsic parameters [fx, fy, ppx, ppy]
            cube_size (int or torch.Tensor): Size of the cropping cube in millimeters
            use_imu (bool): Whether to use IMU data for training/evaluation
            window_before (int): Number of previous frames to include in the IMU sliding window
            window_after (int): Number of subsequent frames to include in the IMU sliding window
            crop_size (tuple): Size (height, width) of the cropped image region
        """
        self.data_root = data_root
        self.kind = kind
        self.use_imu = use_imu
        self.window_before = window_before
        self.window_after = window_after
        cube_size = torch.tensor(cube_size, dtype=torch.float32)
        self.crop_size = crop_size
        xx, yy = np.meshgrid(np.arange(crop_size[0]), np.arange(crop_size[0]))
        padd = np.ones([crop_size[0], crop_size[0]])
        mesh = np.stack((xx, yy, padd), axis=-1).reshape([1, -1, 3])
        self.crop_mesh = torch.from_numpy(mesh).float()  # .cuda()
        if len(cam_paras.shape) == 1:
            self.cam_paras = cam_paras.unsqueeze(0)
        if len(cube_size.shape) == 0:  # If size is a scalar
            self.cube_size = torch.tensor([cube_size, cube_size, cube_size]).unsqueeze(0)

        # Camera rotation matrix
        self.R = torch.tensor(np.array([
            [0.99868261, -0.01849976, -0.04786242],
            [0.01780914, 0.99973163, -0.01481574],
            [0.04812366, 0.01394384, 0.99874405]
        ])).float()

        # Camera translation vector
        self.T = torch.tensor(np.array([39.24702966, 10.84319794, -4.54703251])).float()

        self.label_root = label_root
        self.camera_id = camera_id
        if self.kind == 'train':
            self.subfolders = ['ROM01_Finger_Counting', 'ROM02_Finger_Bending', 'ROM03_Finger_Tapping',
                               'ROM04_Finger_Pinching', 'ROM05_Thumb_Tap_Palm', 'ROM06_Thumb_Tap_Back',
                               'ROM07_Finger_Walking', 'ROM08_Finger_Snapping', 'ROM09_Thumb_Sliding_Palm',
                               'ROM10_Thumb_Sliding_Back', 'ROM11_Finger_Wiggling', 'ROM12_Finger_Crossing']
        if self.kind == 'test':
            self.subfolders = ['ROM13_Finger_Interlocking', 'ROM14_Random_Gestures', 'ROM15_Calibration']

        print("kind:", self.kind)
        print("self.subfolders: ", self.subfolders)
        # print(input())
        self.image_paths = []
        self.j2d_list = []
        self.j3d_list = []
        self.mano_paras_list = []
        self.bbox_list = []
        self.center3d_list = []
        self.acc_list = []
        self.ori_list = []
        self.frame_nums = []

        self._load_data()

    def extract_integer_from_filename(self, filename):
        """Extract the integer part from a filename"""
        match = re.search(r'^\d+', filename)
        if match:
            number_str = match.group()
            return int(number_str)
        else:
            return None

    def affine_grid(self, img, M):
        """Generate an affine grid for image warping"""
        device = img.device
        b, c, h_ori, w_ori = img.size()
        h, w = self.crop_size[0], self.crop_size[1]
        M_inverse = torch.inverse(M).view(b, 1, 3, 3)
        mesh = self.crop_mesh.repeat(b, 1, 1).to(M.device)
        mesh_trans = torch.matmul(M_inverse, mesh.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        mesh_trans = mesh_trans.reshape(b, h * w, 2)
        coeff = torch.Tensor([w_ori, h_ori]).to(device).view(1, 1, 2)
        normal_mesh_trans = (mesh_trans / coeff) * 2 - 1
        return normal_mesh_trans.view(b, h, w, 2)

    def warpPerspective(self, img, M):
        """Apply perspective warping to an image using the given transformation matrix"""
        grid = self.affine_grid(img, M)
        crop_img = F.grid_sample(img, grid, mode='nearest', align_corners=True)
        return crop_img

    def get_original_size(self, M):
        """Get the original image size (height, width)"""
        return 720, 1280

    def inverse_warpPerspective(self, crop_img, M):
        """Apply inverse perspective warping to restore the original image"""
        device = crop_img.device
        b, c, h_crop, w_crop = crop_img.size()
        h_ori, w_ori = self.get_original_size(M)  # Get original image dimensions
        M_inv = torch.inverse(M).view(b, 3, 3)
        x = torch.arange(w_ori).float().to(device)
        y = torch.arange(h_ori).float().to(device)
        mesh_x, mesh_y = torch.meshgrid(x, y)
        mesh = torch.stack([mesh_x, mesh_y, torch.ones_like(mesh_x)], dim=2).unsqueeze(0).repeat(b, 1, 1, 1).to(device)
        mesh = mesh.view(b, -1, 3)
        mesh_trans = torch.matmul(M_inv.unsqueeze(1), mesh.unsqueeze(-1)).squeeze(-1)
        mesh_trans = mesh_trans[:, :, :2]
        mesh_trans = mesh_trans.reshape(b, h_ori, w_ori, 2)
        mesh_trans[..., 0] = (mesh_trans[..., 0] / (w_crop - 1)) * 2 - 1
        mesh_trans[..., 1] = (mesh_trans[..., 1] / (h_crop - 1)) * 2 - 1
        restored_img = F.grid_sample(crop_img, mesh_trans, mode='nearest', align_corners=True)
        return restored_img

    def normalize_img(self, imgD, com, cube):
        """Normalize depth image using center of mass and cube size"""
        z_min = com[:, 2] - cube[:, 2] / 2.
        z_max = com[:, 2] + cube[:, 2] / 2.
        z_max = z_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_min = z_min.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        imgD = torch.where((imgD == -1) | (imgD == 0), z_max, imgD)
        imgD = torch.where(imgD.gt(z_max), z_max, imgD)
        imgD = torch.where(imgD.lt(z_min), z_min, imgD)
        imgD -= com[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        imgD /= (cube[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 2.)
        return imgD

    def denormalize_img(self, normalized_img, com, cube):
        """Denormalize depth image back to original scale"""
        normalized_img = normalized_img.clone()
        normalized_img *= (cube[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 2.)
        normalized_img += com[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_max = com[:, 2] + cube[:, 2] / 2.
        z_max = z_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        normalized_img = torch.where(normalized_img == z_max, z_max, normalized_img)
        z_min = com[:, 2] - cube[:, 2] / 2.
        z_min = z_min.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        normalized_img = torch.where(normalized_img == z_min, z_min, normalized_img)
        original_img = torch.where((normalized_img == -1) | (normalized_img == 0), z_max, normalized_img)
        return original_img

    def JointTrans(self, joint, M, com, cube, paras):
        """Transform joint coordinates using the given matrix and parameters"""
        device = joint.device
        b, j, _ = joint.size()
        joint_uvd = self.points3DToImg(joint, paras)
        joint_trans = torch.cat((joint_uvd[:, :, 0:2], torch.ones([b, j, 1]).to(device)), dim=-1).unsqueeze(-1)
        joint_trans = torch.matmul(M.unsqueeze(1), joint_trans).squeeze(-1)
        joint_uv = joint_trans[:, :, 0:2] / self.crop_size[0] * 2 - 1
        joint_d = (joint_uvd[:, :, 2:] - com.unsqueeze(1)[:, :, 2:]) / (cube.unsqueeze(1)[:, :, 2:] / 2.0)
        return torch.cat((joint_uv, joint_d), dim=-1)

    def points3DToImg(self, joint_xyz, paras):
        """Convert 3D joint coordinates to 2D image coordinates"""
        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * paras[..., 0:1] / (joint_xyz[:, :, 2] + 1e-8) + paras[..., 2:3])
        joint_uvd[:, :, 1] = (joint_xyz[:, :, 1] * paras[..., 1:2] / (joint_xyz[:, :, 2]) + paras[..., 3:4])
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    def comToBounds(self, com, size, paras):
        """Calculate bounding box coordinates from center of mass"""
        zstart = com[:, 2] - size[:, 2] / 2.
        zend = com[:, 2] + size[:, 2] / 2.
        xstart = torch.floor(
            (com[:, 0] * com[:, 2] / paras[:, 0] - size[:, 0] / 2.) / com[:, 2] * paras[:, 0] + 0.5).int()
        xend = torch.floor(
            (com[:, 0] * com[:, 2] / paras[:, 0] + size[:, 0] / 2.) / com[:, 2] * paras[:, 0] + 0.5).int()
        ystart = torch.floor(
            (com[:, 1] * com[:, 2] / paras[:, 1] - size[:, 1] / 2.) / com[:, 2] * paras[:, 1] + 0.5).int()
        yend = torch.floor(
            (com[:, 1] * com[:, 2] / paras[:, 1] + size[:, 1] / 2.) / com[:, 2] * paras[:, 1] + 0.5).int()
        return xstart, xend, ystart, yend, zstart, zend

    def Offset2Trans(self, xstart, xend, ystart, yend):
        """Generate transformation matrix from bounding box coordinates"""
        device = xstart.device
        b = xstart.size(0)
        wb = (xend - xstart)
        hb = (yend - ystart)

        sz0 = torch.where(wb.gt(hb), torch.ones_like(wb).to(device) * self.crop_size[0],
                          (wb * self.crop_size[0] / hb).int())
        sz1 = torch.where(wb.gt(hb), (hb * self.crop_size[0] / wb).int(),
                          torch.ones_like(wb).to(device) * self.crop_size[1])

        s = torch.where(wb.gt(hb), self.crop_size[0] / wb, self.crop_size[1] / hb)

        trans = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        trans[:, 0, 2] = -xstart
        trans[:, 1, 2] = -ystart
        scale = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        scale[:, 0, 0] = s
        scale[:, 1, 1] = s

        xstart = (torch.floor(self.crop_size[0] / 2. - sz0 / 2.)).int()
        ystart = (torch.floor(self.crop_size[1] / 2. - sz1 / 2.)).int()
        off = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).to(device)
        off[:, 0, 2] = xstart
        off[:, 1, 2] = ystart

        M = torch.matmul(off, torch.matmul(scale, trans))
        return M

    def _load_data(self):
        """Load dataset from files"""
        for subfolder in self.subfolders:
            key_root = os.path.join(self.label_root, subfolder, "joint.json")
            image_root = os.path.join(self.data_root, subfolder, self.camera_id, "rgb")
            bbox_root = os.path.join(self.data_root, subfolder, self.camera_id, "bbox.json")
            mano_paras_root = os.path.join(self.label_root, subfolder, "mano.json")
            center_root = os.path.join(self.data_root, subfolder, self.camera_id, "center.json")
            center_all = json.load(open(center_root))

            key_all_data = json.load(open(key_root))
            # bbox_all_data = json.load(open(bbox_root))
            mano_paras_all = json.load(open(mano_paras_root))
            image_files = glob.glob(os.path.join(image_root, "*.jpg")) + glob.glob(os.path.join(image_root, "*.png"))
            # print(image_root)
            if self.use_imu:
                imu_root = os.path.join(self.data_root, subfolder, "IMU.json")
                imu_all_data = json.load(open(imu_root))

            for image_file in image_files:
                image_name = os.path.basename(image_file)
                # if image_name not in key_all_data["j2d"][self.camera_id]:
                if image_name not in key_all_data:
                    print("Missing label for this data:", subfolder + "/" + image_name)
                    continue

                frame_num = self.extract_integer_from_filename(image_name)
                self.frame_nums.append(frame_num)
                # self.camera_id
                # joint2d = np.array(key_all_data["j2d"]["camera0"][image_name], dtype=np.float32).reshape(21, 2)
                # joint3d = np.array(key_all_data["j3d"][image_name], dtype=np.float32).reshape(21, 3)
                mano_paras = np.array(mano_paras_all[image_name], dtype=np.float32)
                # bbox = np.array(bbox_all_data[image_name], dtype=np.float32)
                center3d = np.array(center_all[image_name], dtype=np.float32)

                if self.use_imu:
                    try:
                        imudata = imu_all_data[str(frame_num)]["Frames_Data"]
                    except KeyError as e:
                        print(f"KeyError: {e} in file {image_file}")
                        print(f"Available keys in imu_all_data: {list(imu_all_data.keys())[-1]}")
                        continue

                    accc = np.array([[imu['acc_x'], imu['acc_y'], imu['acc_z']] for imu in imudata])
                    q = np.array([[imu['q_0'], imu['q_1'], imu['q_2'], imu['q_3']] for imu in imudata])
                    oric = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()

                    ACC_MIN = -5.0
                    ACC_MAX = 5.0
                    Q_MIN = -1.0
                    Q_MAX = 1.0

                    for sensor in range(accc.shape[0]):
                        for dim in range(3):  # X, Y, Z dimensions
                            if not (ACC_MIN <= accc[sensor, dim] <= ACC_MAX):
                                if prev_acc is not None:
                                    accc[sensor, dim] = prev_acc[sensor, dim]  # Replace with previous frame's value
                                else:
                                    print(
                                        f"Warning: First frame's IMU {sensor + 1} acceleration out of bounds. No previous frame to replace.")

                    # Check if quaternion components are out of range
                    for sensor in range(q.shape[0]):
                        for dim in range(4):  # q0, q1, q2, q3 components
                            if not (Q_MIN <= q[sensor, dim] <= Q_MAX):
                                if prev_q is not None:
                                    q[sensor, dim] = prev_q[sensor, dim]  # Replace with previous frame's value
                                else:
                                    print(
                                        f"Warning: First frame's IMU {sensor + 1} quaternion out of bounds. No previous frame to replace.")

                    prev_acc = accc.copy()
                    prev_q = q.copy()

                    "Align IMU data to pose"
                    accw = torch.from_numpy(accc).float()
                    oriw = torch.from_numpy(oric).float()
                    quat_dim = 6
                    Rrw = mano_paras[0:quat_dim]
                    Rrw = torch.from_numpy(Rrw).float()
                    Rrw = rotation_6d_to_matrix(Rrw)
                    accr = Rrw.unsqueeze(0).matmul(accw.unsqueeze(-1))
                    orir = Rrw.unsqueeze(0).matmul(oriw)
                    accc = accr.squeeze(-1)
                    oric = orir.squeeze(-1)

                    self.acc_list.append(accc)
                    self.ori_list.append(oric)

                self.center3d_list.append(center3d)
                self.image_paths.append(image_file)
                self.mano_paras_list.append(mano_paras)

        self.mano_paras = np.stack(self.mano_paras_list)
        self.center3d = np.stack(self.center3d_list)
        if self.use_imu:
            self.accr = np.stack(self.acc_list)
            self.orir = np.stack(self.ori_list)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a sample from the dataset by index"""
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        orimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(orimage).permute(2, 0, 1).float()
        mano_paras = torch.from_numpy(self.mano_paras[idx]).float()
        center3d = torch.from_numpy(self.center3d[idx]).float()

        quat_dim = 6  # Rotation dimension
        quat = mano_paras[0:quat_dim].clone().detach().unsqueeze(0)  # Extract rotation parameters
        theta = mano_paras[quat_dim:quat_dim + 45].clone().detach().unsqueeze(0)  # Hand pose parameters
        beta = mano_paras[quat_dim + 45:quat_dim + 45 + 10].clone().detach().unsqueeze(0)  # Shape parameters
        cam = mano_paras[quat_dim + 45 + 10:quat_dim + 45 + 10 + 4].clone().detach().unsqueeze(0)  # Camera parameters

        # Get MANO hand model vertices and joints
        hand_verts, all_hand_joints, pose, shape, cam = mano_layer.get_mano_vertices(quat, theta, beta, cam,
                                                                                     global_scale=1 / 125)
        hand_joints = all_hand_joints.clone().squeeze(0)
        trans_center3d = (self.R @ center3d).squeeze(-1) + self.T
        hand_joints = (hand_joints.to(trans_center3d.device) + 1) / 2 * self.cube_size + trans_center3d
        hand_verts = (hand_verts.to(trans_center3d.device) + 1) / 2 * self.cube_size + trans_center3d
        j3d = hand_joints

        center3d = trans_center3d.unsqueeze(0)
        center2d = self.points3DToImg(center3d.unsqueeze(1), self.cam_paras).squeeze(1)
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(center2d, self.cube_size, self.cam_paras)
        M = self.Offset2Trans(xstart, xend, ystart, yend)
        cropped_img = self.warpPerspective(image.unsqueeze(0), M)
        noraml_img = (cropped_img.squeeze(0) / 255.0) * 2 - 1

        noraml_joint_uvd = self.JointTrans(j3d.unsqueeze(0), M, center2d, self.cube_size, self.cam_paras)
        noraml_joint_xyz = (j3d.unsqueeze(0) - center3d.unsqueeze(1)) / self.cube_size.unsqueeze(1) * 2
        noraml_vert_xyz = (hand_verts.unsqueeze(0) - center3d.unsqueeze(1)) / self.cube_size.unsqueeze(1) * 2
        center3d = center3d.squeeze(0)
        center2d = center2d.squeeze(0)
        M = M.squeeze(0)
        noraml_joint_uvd = noraml_joint_uvd.squeeze(0)
        noraml_joint_xyz = noraml_joint_xyz.squeeze(0)

        if self.use_imu:
            "IMU sliding window sequence padding"
            current_frame_idx = idx
            start_idx = max(0, current_frame_idx - self.window_before)
            end_idx = min(len(self.accr), current_frame_idx + self.window_after + 1)
            window_acc = self.accr[start_idx:end_idx]
            window_ori = self.orir[start_idx:end_idx]
            window_size = self.window_before + self.window_after + 1
            actual_window_size = len(window_acc)

            if actual_window_size < window_size:
                pad_before = max(0, self.window_before - (current_frame_idx - start_idx))
                pad_after = max(0, self.window_after - (end_idx - current_frame_idx - 1))

                pad_before = max(0, pad_before)
                pad_after = max(0, pad_after)

                if window_acc.ndim == 3:
                    window_acc = np.pad(window_acc, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
                    window_ori = np.pad(window_ori, ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)), mode='edge')
                elif window_acc.ndim == 2:
                    window_acc = np.pad(window_acc, ((pad_before, pad_after), (0, 0)), mode='edge')
                    window_ori = np.pad(window_ori, ((pad_before, pad_after), (0, 0)), mode='edge')
                else:
                    pad_width = [(pad_before, pad_after)] * window_acc.ndim
                    window_acc = np.pad(window_acc, pad_width, mode='edge')
                    window_ori = np.pad(window_ori, pad_width, mode='edge')

            window_acc = torch.tensor(window_acc, dtype=torch.float32)
            window_ori = torch.tensor(window_ori, dtype=torch.float32)

            return {
                'orimage': orimage,
                'image': noraml_img,
                'j2d': noraml_joint_uvd,
                'j3d': noraml_joint_xyz,
                'vert': noraml_vert_xyz,
                'mano_paras': mano_paras,
                'center3d': center3d,
                'center2d': center2d,
                'M': M,
                'acc_w': window_acc,
                'ori_w': window_ori
            }

        else:
            return {
                'orimage': orimage,
                'image': noraml_img,
                'j2d': noraml_joint_uvd,
                'j3d': noraml_joint_xyz,
                'vert': noraml_vert_xyz,
                'mano_paras': mano_paras,
                'center3d': center3d,
                'center2d': center2d,
                'M': M
            }


if __name__ == "__main__":
    # List of 15 subjects
    all_subjects = [
        'subject1', 'subject2', 'subject3', 'subject4', 'subject5',
        'subject6', 'subject7', 'subject8', 'subject9', 'subject10',
        'subject11', 'subject12', 'subject13', 'subject14', 'subject15'
    ]

    train_datasets = []
    for subject in all_subjects:
        print("subject:", subject)
        data_root_dir = f"/root/multidata/{subject}_imu/"
        label_root_dir = f"/root/annotation/{subject}/"
        camera_id = "camera0"
        cam_paras = json.load(open(data_root_dir + r'camera_params.json'))
        intrinsics = np.array([cam_paras[camera_id]['intrinsic']])
        extrinsics = np.array([cam_paras[camera_id]['extrinsic']])
        cam_paras = torch.tensor([
            intrinsics.squeeze(0)[0][0],  # fx
            intrinsics.squeeze(0)[1][1],  # fy
            intrinsics.squeeze(0)[0][2],  # ppx
            intrinsics.squeeze(0)[1][2]  # ppy
        ], dtype=torch.float32)

        # Create dataset for each subject
        dataset = SelfDataset(
            data_root=data_root_dir,
            label_root=label_root_dir,
            camera_id=camera_id,
            kind="train",
            cam_paras=cam_paras,
            use_imu=False,
            window_before=10,
            window_after=0
        )
        train_datasets.append(dataset)

    # Combine all subject datasets into one
    train_dataset = ConcatDataset(train_datasets)
    # Create data loader for training
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)