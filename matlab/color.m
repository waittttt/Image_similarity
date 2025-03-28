% 文件夹路径（可以根据需要调整）
folder_path = './';  % 当前文件夹，如果图像在其他位置，修改为相应路径

% 遍历 image1.png 到 image10.png
for i = 1:10
    % 构建文件名
    img_name = sprintf('image%d.png', i);
    
    % 读取图像
    img = imread(fullfile(folder_path, img_name));
    
    % 将图像转换为double类型以便处理
    img_double = im2double(img);
    
    % 统一调整亮度和对比度
    % 假设我们将图像亮度提高20%并且增强对比度
    img_adjusted = img_double * 1.2;  % 增加亮度
    img_adjusted = imadjust(img_adjusted, [0.2 0.8], [0 1]);  % 增强对比度
    
    % 将调整后的图像转换回uint8类型
    img_adjusted = im2uint8(img_adjusted);
    
    % 显示原图和调整后的图像
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_adjusted), title(['Adjusted Image ', num2str(i)]);
    
    % 保存调整后的图像
    adjusted_img_name = sprintf('adjusted_image%d.png', i);  % 保存为 adjusted_image1.png 等
    imwrite(img_adjusted, fullfile(folder_path, adjusted_img_name));
end
