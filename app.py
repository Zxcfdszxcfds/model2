import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="图像滤波与频域变换平台", layout="wide")
st.title("📷 图像滤波与频域变换实验（A2作业）")

# ---------------------- 通用图片上传 ----------------------
uploaded_file = st.file_uploader("上传一张图片（支持JPG/PNG）", type=["jpg", "png"], key="main_upload")

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # 转为灰度图，简化计算
    img_np = np.array(img)
    h, w = img_np.shape
    st.image(img_np, caption="原图", use_column_width=True, channels="gray")

    # ---------------------- 1. 空间图像滤波器对比 ----------------------
    st.header("1. 空间滤波器对比（Box/Gaussian/Median/Sobel）")
    filter_type = st.selectbox("选择滤波器类型", 
                               ["Box均值滤波", "Gaussian高斯滤波", "Median中值滤波", "Sobel边缘检测"],
                               key="filter_select")
    kernel_size = st.slider("核大小", min_value=3, max_value=11, value=3, step=2, key="kernel_slider")

    if st.button("应用滤波器", key="apply_filter_btn"):
        with st.spinner("滤波计算中..."):
            pad = kernel_size // 2
            padded_img = np.pad(img_np, pad, mode="edge")  # 边缘填充，避免黑边
            result = np.zeros_like(img_np, dtype=np.float32)

            if filter_type == "Box均值滤波":
                # 均值滤波核
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                for i in range(pad, h + pad):
                    for j in range(pad, w + pad):
                        region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
                        result[i-pad, j-pad] = np.sum(region * kernel)

            elif filter_type == "Gaussian高斯滤波":
                # 高斯核生成
                sigma = 1.0
                x, y = np.mgrid[-pad:pad+1, -pad:pad+1]
                kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                kernel /= kernel.sum()  # 归一化
                for i in range(pad, h + pad):
                    for j in range(pad, w + pad):
                        region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
                        result[i-pad, j-pad] = np.sum(region * kernel)

            elif filter_type == "Median中值滤波":
                for i in range(pad, h + pad):
                    for j in range(pad, w + pad):
                        region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
                        result[i-pad, j-pad] = np.median(region)

            elif filter_type == "Sobel边缘检测":
                # Sobel梯度核
                kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                gx, gy = np.zeros_like(img_np, dtype=np.float32), np.zeros_like(img_np, dtype=np.float32)
                for i in range(1, h+1):
                    for j in range(1, w+1):
                        region = padded_img[i-1:i+2, j-1:j+2]
                        gx[i-1, j-1] = np.sum(region * kx)
                        gy[i-1, j-1] = np.sum(region * ky)
                result = np.sqrt(gx**2 + gy**2)

            # 显示结果对比
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(result, cmap="gray")
            axes[1].set_title(f"{filter_type} 结果")
            axes[1].axis("off")
            st.pyplot(fig)

    # ---------------------- 2. 图像梯度方向演示 ----------------------
    st.header("2. 图像梯度方向计算（局部区域演示）")
    # 让用户选择局部区域（简化为中心区域）
    st.info("默认选取图像中心区域计算梯度，可直接观察方向分布")
    if st.button("计算梯度方向", key="grad_btn"):
        with st.spinner("计算梯度中..."):
            # Sobel梯度计算
            kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            pad = 1
            padded = np.pad(img_np, pad, mode="edge")
            gx, gy = np.zeros_like(img_np, dtype=np.float32), np.zeros_like(img_np, dtype=np.float32)
            
            for i in range(pad, h+pad):
                for j in range(pad, w+pad):
                    region = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                    gx[i-pad, j-pad] = np.sum(region * kx)
                    gy[i-pad, j-pad] = np.sum(region * ky)
            
            # 梯度幅值与方向
            mag = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx) * 180 / np.pi  # 转为角度
            
            # 取中心局部区域
            center_h, center_w = h//2, w//2
            local_region = mag[center_h-50:center_h+50, center_w-50:center_w+50]
            local_dir = direction[center_h-50:center_h+50, center_w-50:center_w+50]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("原图（中心区域为演示区）")
            axes[0].axis("off")
            axes[1].imshow(local_region, cmap="gray")
            axes[1].set_title("局部区域梯度幅值")
            axes[1].axis("off")
            axes[2].imshow(local_dir, cmap="hsv")
            axes[2].set_title("局部区域梯度方向（角度）")
            axes[2].axis("off")
            st.pyplot(fig)

    # ---------------------- 3. 频域图像滤波（傅里叶变换） ----------------------
    st.header("3. 频域变换与频谱分析（旋转/平移/缩放）")
    if st.button("计算原图频谱", key="fft_btn"):
        with st.spinner("傅里叶变换计算中..."):
            # 傅里叶变换与频谱
            fft = np.fft.fft2(img_np)
            fft_shift = np.fft.fftshift(fft)
            spectrum = 20 * np.log(np.abs(fft_shift) + 1)  # 对数增强，方便显示
            
            # 逆变换验证
            ifft_img = np.fft.ifft2(np.fft.ifftshift(fft_shift)).real

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(spectrum, cmap="gray")
            axes[1].set_title("傅里叶频谱图")
            axes[1].axis("off")
            axes[2].imshow(ifft_img, cmap="gray")
            axes[2].set_title("逆傅里叶变换结果")
            axes[2].axis("off")
            st.pyplot(fig)

    # 频谱变化对比（旋转/平移/缩放）
    if st.button("对比图像变换后的频谱变化", key="fft_compare_btn"):
        with st.spinner("对比计算中..."):
            # 1. 原图频谱
            fft_orig = np.fft.fft2(img_np)
            spec_orig = 20 * np.log(np.abs(np.fft.fftshift(fft_orig)) + 1)
            
            # 2. 旋转90度后的频谱
            img_rot = np.rot90(img_np)
            fft_rot = np.fft.fft2(img_rot)
            spec_rot = 20 * np.log(np.abs(np.fft.fftshift(fft_rot)) + 1)
            
            # 3. 平移后的频谱（右移100像素）
            img_shift = np.roll(img_np, 100, axis=1)
            fft_shift = np.fft.fft2(img_shift)
            spec_shift = 20 * np.log(np.abs(np.fft.fftshift(fft_shift)) + 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(spec_orig, cmap="gray")
            axes[0].set_title("原图频谱")
            axes[0].axis("off")
            axes[1].imshow(spec_rot, cmap="gray")
            axes[1].set_title("旋转后频谱（旋转不变性）")
            axes[1].axis("off")
            axes[2].imshow(spec_shift, cmap="gray")
            axes[2].set_title("平移后频谱（相位变化，幅度不变）")
            axes[2].axis("off")
            st.pyplot(fig)

st.markdown("---")
st.caption("模式识别与图像处理 - A2作业平台 | 无OpenCV依赖，Streamlit可直接部署")
