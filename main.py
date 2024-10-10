import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取相机参数
def load_camera_parameters(json_file):
    with open(json_file, 'r') as f:
        cameras = json.load(f)
    return cameras

# 读取相机内外参数
def get_camera_matrices(cameras):
    K_matrices = []
    R_matrices = []
    t_vectors = []

    for camera in cameras:
        # 提取内参矩阵 K
        K = np.array(camera['intrinsic'], dtype=np.float64)
        K_matrices.append(K)

        # 提取外参矩阵（旋转和平移）
        extrinsic = np.array(camera['extrinsic'], dtype=np.float64)
        R = extrinsic[:, :3]  # 旋转部分
        t = extrinsic[:, 3]   # 平移部分

        R_matrices.append(R)
        t_vectors.append(t)

    return K_matrices, R_matrices, t_vectors

# 绘制带有特征点的 2D 图像
def draw_keypoints(image, keypoints):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return image_with_keypoints



# 加载三幅图像
image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('image3.png', cv2.IMREAD_GRAYSCALE)

# 提取相机参数
cameras = load_camera_parameters('extrinsic.ikdc.20240927.1.json')
K_matrices, R_matrices, t_vectors = get_camera_matrices(cameras)

# 初始化 SIFT 特征提取器
sift = cv2.SIFT_create()

# 提取关键点和描述符
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
kp3, des3 = sift.detectAndCompute(image3, None)

# 使用 BFMatcher 进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches12 = bf.match(des1, des2)
matches13 = bf.match(des1, des3)

# 按距离排序匹配项
matches12 = sorted(matches12, key=lambda x: x.distance)
matches13 = sorted(matches13, key=lambda x: x.distance)

# 提取匹配点的坐标
pts1_12 = np.float32([kp1[m.queryIdx].pt for m in matches12])
pts2_12 = np.float32([kp2[m.trainIdx].pt for m in matches12])

pts1_13 = np.float32([kp1[m.queryIdx].pt for m in matches13])
pts3_13 = np.float32([kp3[m.trainIdx].pt for m in matches13])

# 计算基础矩阵和本质矩阵（使用 RANSAC 排除外点）
F12, mask12 = cv2.findFundamentalMat(pts1_12, pts2_12, cv2.FM_RANSAC)
E12, _ = cv2.findEssentialMat(pts1_12, pts2_12, K_matrices[0], method=cv2.RANSAC)

F13, mask13 = cv2.findFundamentalMat(pts1_13, pts3_13, cv2.FM_RANSAC)
E13, _ = cv2.findEssentialMat(pts1_13, pts3_13, K_matrices[0], method=cv2.RANSAC)

# 恢复相机姿态（旋转矩阵和平移向量）
_, R12, t12, _ = cv2.recoverPose(E12, pts1_12, pts2_12, K_matrices[0])
_, R13, t13, _ = cv2.recoverPose(E13, pts1_13, pts3_13, K_matrices[0])

# 构建相机投影矩阵
M1 = np.hstack((K_matrices[0], np.zeros((3, 1))))  # 第一相机的投影矩阵 [K|0]
M2 = np.hstack((R12 @ K_matrices[0], t12))         # 第二相机的投影矩阵
M3 = np.hstack((R13 @ K_matrices[0], t13))         # 第三相机的投影矩阵

# 三角测量图像1和图像2的点云
pts_4d_homogeneous_12 = cv2.triangulatePoints(M1, M2, pts1_12.T, pts2_12.T)
pts_3d_12 = pts_4d_homogeneous_12 / pts_4d_homogeneous_12[3]  # 齐次坐标转3D坐标
pts_3d_12 = pts_3d_12[:3].T

# 三角测量图像1和图像3的点云
pts_4d_homogeneous_13 = cv2.triangulatePoints(M1, M3, pts1_13.T, pts3_13.T)
pts_3d_13 = pts_4d_homogeneous_13 / pts_4d_homogeneous_13[3]
pts_3d_13 = pts_3d_13[:3].T

# 合并两组点云
pts_3d_combined = np.vstack((pts_3d_12, pts_3d_13))

# 绘制 3D 点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 获取 X, Y, Z 坐标
x_vals = pts_3d_combined[:, 0]
y_vals = pts_3d_combined[:, 1]
z_vals = pts_3d_combined[:, 2]

ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


# 显示带有特征点的图像
image1_with_kp = draw_keypoints(image1, kp1)
image2_with_kp = draw_keypoints(image2, kp2)
image3_with_kp = draw_keypoints(image3, kp3)

# 显示图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image1_with_kp, cmap='gray')
plt.title('Image 1 with Keypoints')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image2_with_kp, cmap='gray')
plt.title('Image 2 with Keypoints')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image3_with_kp, cmap='gray')
plt.title('Image 3 with Keypoints')
plt.axis('off')

plt.tight_layout()
plt.show()