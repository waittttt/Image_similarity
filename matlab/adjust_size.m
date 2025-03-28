% 文件夹路径（可以根据需要调整）
folder_path = './';  % 当前文件夹，如果图像在其他位置，修改为相应路径

% 设置统一的目标大小（例如 512x512）
target_size = [1024, 1024];  % 目标大小，行和列分别为高度和宽度

% 遍历 image1.png 到 image10.png
for i = 1:10
    % 构建文件名
    img_name = sprintf('image%d.png', i);
    
    % 读取图像
    img = imread(fullfile(folder_path, img_name));
    
    % 调整图像大小
    img_resized = imresize(img, target_size);  % 将图像调整为目标大小
    
    % 显示原图和调整后的图像
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_resized), title(['Resized Image ', num2str(i)]);
    
    % 保存调整后的图像
    resized_img_name = sprintf('resized_image%d.png', i);  % 保存为 resized_image1.png 等
    imwrite(img_resized, fullfile(folder_path, resized_img_name));
end
