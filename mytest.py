import numpy as np
import os
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from modules.xfeat import XFeat

# -------------------------- 1. 设备与模型初始化 --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")

try:
    xfeat = XFeat().to(device)
    print("XFeat模型初始化成功")
except Exception as e:
    raise RuntimeError(f"XFeat模型初始化失败: {str(e)}") from e


# -------------------------- 2. 工具函数定义 --------------------------
def load_image(image_path):
    """加载RGB图像，返回numpy数组"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片路径不存在: {image_path}")
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
        raise ValueError(f"图像需为3通道RGB格式，当前格式: {img_np.shape}")
    return img_np


def draw_matches_v2(img1, img2, mkpts1, mkpts2, title_suffix=""):
    """绘制特征点匹配结果"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    mkpts1 = np.asarray(mkpts1)
    mkpts2 = np.asarray(mkpts2)
    for (x1, y1), (x2, y2) in zip(mkpts1, mkpts2):
        x1_int, y1_int = int(round(x1)), int(round(y1))
        x2_int, y2_int = int(round(x2)), int(round(y2))
        x2_int += w1
        cv2.line(canvas, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 1)
        cv2.circle(canvas, (x1_int, y1_int), 3, (255, 0, 0), -1)
        cv2.circle(canvas, (x2_int, y2_int), 3, (0, 0, 255), -1)

    text = f"匹配点数量: {len(mkpts1)} {title_suffix}"
    cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def ransac_geometric_filter(mkpts1, mkpts2, reproj_thr=4.0):
    """RANSAC几何精筛"""
    if len(mkpts1) < 4 or len(mkpts2) < 4:
        print(f"匹配点数量不足（{len(mkpts1)}个），跳过RANSAC筛选")
        return mkpts1, mkpts2

    pts1 = np.asarray(mkpts1, dtype=np.float32)
    pts2 = np.asarray(mkpts2, dtype=np.float32)

    try:
        H, inliers = cv2.findHomography(
            pts1, pts2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=reproj_thr,
            maxIters=700,
            confidence=0.995
        )
    except Exception as e:
        print(f"RANSAC计算失败: {str(e)}，跳过RANSAC筛选")
        return mkpts1, mkpts2

    inliers = inliers.flatten()
    filtered_mkpts1 = mkpts1[inliers == 1]
    filtered_mkpts2 = mkpts2[inliers == 1]

    print(f"\n=== RANSAC筛选结果 ===")
    print(f"筛选前匹配点数量: {len(mkpts1)}")
    print(f"筛选后匹配点数量: {len(filtered_mkpts1)}")
    print(f"误匹配剔除率: {100 * (1 - len(filtered_mkpts1) / len(mkpts1)):.2f}%")

    return filtered_mkpts1, filtered_mkpts2


# -------------------------- 3. 主流程（三种匹配方式对比） --------------------------
if __name__ == "__main__":
    # 1. 图片路径
    image1_path = "E:\\Keep\\vision\\accelerated_features-main\\pic\\test.png"
    image2_path = "E:\\Keep\\vision\\accelerated_features-main\\pic\\dr4.png"

    # 2. 加载图片（保留原始图像数据，用于XFeat原生匹配）
    print(f"\n正在加载图片...")
    try:
        img1 = load_image(image1_path)  # 原始图像（numpy数组，HWC格式）

        img2 = load_image(image2_path)  # 原始图像（numpy数组，HWC格式）
        print(f"图片加载成功，尺寸: img1={img1.shape}, img2={img2.shape}")
    except Exception as e:
        raise RuntimeError(f"图片加载失败: {str(e)}") from e

    # 3. 提取特征（仅为lighterglue匹配服务）
    print(f"\n正在提取特征点（为lighterglue准备）...")
    with torch.no_grad():
        try:
            output0 = xfeat.detectAndCompute(img1, top_k=4096)[0]  # 特征字典
            output1 = xfeat.detectAndCompute(img2, top_k=4096)[0]  # 特征字典
            output0["image_size"] = (img1.shape[1], img1.shape[0])
            output1["image_size"] = (img2.shape[1], img2.shape[0])
            print(f"特征提取成功，特征点数量: img1={len(output0['keypoints'])}, img2={len(output1['keypoints'])}")
        except Exception as e:
            raise RuntimeError(f"特征提取失败: {str(e)}") from e

    # -------------------------- 4. XFeat原生匹配（核心修正：输入原始图像而非特征字典） --------------------------
    print(f"\n正在进行XFeat原生特征匹配...")
    with torch.no_grad():
        try:
            # 关键修正：传入原始图像（img1, img2）而非特征字典（output0, output1）
            xfeat_raw_match = xfeat.match(img1, img2)  # 原生匹配直接用图像

            # 处理返回值（假设返回tuple：(图1匹配点, 图2匹配点)）
            if isinstance(xfeat_raw_match, tuple) and len(xfeat_raw_match) >= 2:
                mkpts_0_xfeat, mkpts_1_xfeat = xfeat_raw_match[0], xfeat_raw_match[1]
            else:
                raise ValueError("XFeat原生匹配返回格式异常，需为含2个元素的tuple")

            # 格式转换与验证
            mkpts_0_xfeat = np.asarray(mkpts_0_xfeat)
            mkpts_1_xfeat = np.asarray(mkpts_1_xfeat)
            if len(mkpts_0_xfeat) == 0 or len(mkpts_1_xfeat) == 0:
                raise ValueError("XFeat原生匹配未找到任何匹配点")

            print(f"XFeat原生匹配成功，匹配点数量: {len(mkpts_0_xfeat)}")
        except Exception as e:
            # 尝试备用接口（同样传入原始图像）
            print(f"接口 match 调用失败，尝试备用接口 match_xfeat...")
            try:
                # 关键修正：同样传入原始图像
                mkpts_0_xfeat, mkpts_1_xfeat = xfeat.match_xfeat(img1, img2)
                mkpts_0_xfeat = np.asarray(mkpts_0_xfeat)
                mkpts_1_xfeat = np.asarray(mkpts_1_xfeat)
                if len(mkpts_0_xfeat) == 0:
                    raise ValueError("备用接口仍未找到匹配点")
                print(f"备用接口 match_xfeat 匹配成功，匹配点数量: {len(mkpts_0_xfeat)}")
            except Exception as e2:
                raise RuntimeError(f"XFeat原生匹配彻底失败: {str(e2)}") from e2

    # -------------------------- 5. XFeat + lighterglue 匹配 --------------------------
    print(f"\n正在进行XFeat + lighterglue 匹配...")
    with torch.no_grad():
        try:
            match_result = xfeat.match_lighterglue(output0, output1)
            print(f"match_lighterglue返回值数量: {len(match_result)}")
            mkpts_0_lighter, mkpts_1_lighter = match_result[0], match_result[1]

            mkpts_0_lighter = np.asarray(mkpts_0_lighter)
            mkpts_1_lighter = np.asarray(mkpts_1_lighter)
            if len(mkpts_0_lighter) == 0:
                raise ValueError("lighterglue未找到任何匹配点")

            print(f"XFeat + lighterglue 匹配成功，匹配点数量: {len(mkpts_0_lighter)}")
        except Exception as e:
            raise RuntimeError(f"XFeat + lighterglue 匹配失败: {str(e)}") from e

    # -------------------------- 6. RANSAC几何精筛 --------------------------
    print(f"\n正在进行RANSAC几何精筛...")
    mkpts_0_filtered, mkpts_1_filtered = ransac_geometric_filter(
        mkpts_0_lighter, mkpts_1_lighter, reproj_thr=20.0
    )

    # -------------------------- 7. 三种匹配结果可视化对比 --------------------------
    print(f"\n正在生成匹配结果对比图...")
    try:
        # 绘制三张对比图
        canvas_xfeat = draw_matches_v2(
            img1, img2, mkpts_0_xfeat, mkpts_1_xfeat, "(XFeat原生匹配)"
        )
        canvas_lighter = draw_matches_v2(
            img1, img2, mkpts_0_lighter, mkpts_1_lighter, "(XFeat+lighterglue)"
        )
        canvas_filtered = draw_matches_v2(
            img1, img2, mkpts_0_filtered, mkpts_1_filtered, "(+RANSAC精筛)"
        )

        # 显示对比结果（1行3列布局）
        plt.figure(figsize=(30, 10))
        plt.subplot(131), plt.imshow(canvas_xfeat), plt.title("XFeat原生匹配", fontsize=16), plt.axis('off')
        plt.subplot(132), plt.imshow(canvas_lighter), plt.title("XFeat+lighterglue", fontsize=16), plt.axis('off')
        plt.subplot(133), plt.imshow(canvas_filtered), plt.title("+RANSAC精筛", fontsize=16), plt.axis('off')
        plt.tight_layout(pad=3.0), plt.show()

        # 保存结果
        save_dir = "match_results"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "xfeat_raw.jpg"), cv2.cvtColor(canvas_xfeat, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "xfeat_lighter.jpg"), cv2.cvtColor(canvas_lighter, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "xfeat_lighter_ransac.jpg"),
                    cv2.cvtColor(canvas_filtered, cv2.COLOR_RGB2BGR))
        print(f"结果已保存到: {os.path.abspath(save_dir)}")
    except Exception as e:
        raise RuntimeError(f"可视化失败: {str(e)}") from e

    print(f"\n所有流程完成！")
