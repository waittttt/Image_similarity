% 文件夹路径（可以根据需要调整）
folder_path = './';  % 当前文件夹，如果图像在其他位置，修改为相应路径

% 遍历 image1.png 到 image10.png
for i = 1:10
    % 构建文件名
    img_name = sprintf('image%d.png', i);
    
    % 读取图像
    img = imread(fullfile(folder_path, img_name));
    
    % 添加高斯噪声
    noisy_img = imnoise(img, 'gaussian', 0, 0.05);
    
    % 显示原图和加噪声后的图像
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(noisy_img), title(['Image with Gaussian Noise ', num2str(i)]);
    
    % 保存加噪声后的图像
    noisy_img_name = sprintf('noisy_image%d.png', i);  % 保存为 noisy_image1.png 等
    imwrite(noisy_img, fullfile(folder_path, noisy_img_name));
end
