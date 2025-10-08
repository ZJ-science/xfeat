import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    """加载图像并保持原始色彩"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像路径不存在: {image_path}")

    # 用PIL加载图像保持RGB格式
    img_pil = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img_pil)

    # 同时创建灰度图用于特征提取
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray


def draw_matches(img1, img2, kp1, kp2, matches):
    """
    绘制匹配结果，保持原图色彩不变，并用线连接匹配的特征点
    """
    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 创建拼接图像的画布
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1 + w2] = img2

    # 绘制匹配线和特征点
    for match in matches:
        # 获取特征点坐标
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(kp2[match.trainIdx].pt[0]) + w1, int(kp2[match.trainIdx].pt[1]))

        # 绘制连接线（绿色）
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        # 绘制特征点（蓝色）
        cv2.circle(combined_img, pt1, 3, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(combined_img, pt2, 3, (255, 0, 0), -1, cv2.LINE_AA)

    # 添加匹配点数量文字
    text = f"匹配点数量: {len(matches)}"
    cv2.putText(combined_img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combined_img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return combined_img


def sift_flann_match(img1_path, img2_path):
    """使用SIFT和FLANN进行特征点匹配"""
    # 加载图像
    img1_rgb, img1_gray = load_image(img1_path)
    img2_rgb, img2_gray = load_image(img2_path)

    # 初始化SIFT检测器
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        raise ImportError("请安装带contrib的OpenCV: pip install opencv-contrib-python")

    # 检测特征点并计算描述符
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    print(f"检测到的特征点数量: 图1={len(kp1)}, 图2={len(kp2)}")

    # 初始化FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 增加检查次数可提高精度但减慢速度

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用Lowe's比率测试筛选良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"筛选后的匹配点数量: {len(good_matches)}")

    # 使用RANSAC进一步筛选匹配点
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应矩阵并筛选内点
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()

        # 只保留内点
        ransac_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
        print(f"RANSAC筛选后的匹配点数量: {len(ransac_matches)}")
    else:
        ransac_matches = good_matches
        print("匹配点太少，无法进行RANSAC筛选")

    # 绘制匹配结果
    result_image = draw_matches(img1_rgb, img2_rgb, kp1, kp2, ransac_matches)

    # 显示结果
    plt.figure(figsize=(18, 10))
    plt.imshow(result_image)
    plt.title('SIFT特征点匹配结果', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 保存结果
    save_dir = "matching_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sift_matches.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"匹配结果已保存至: {save_path}")

    return result_image


if __name__ == "__main__":
    # 替换为你的两张相同照片的路径
    image1_path = "E:\\Keep\\vision\\accelerated_features-main\\pic\\test.png"
    image2_path = "E:\\Keep\\vision\\accelerated_features-main\\pic\\dr3.png"

    try:
        sift_flann_match(image1_path, image2_path)
    except Exception as e:
        print(f"发生错误: {str(e)}")
