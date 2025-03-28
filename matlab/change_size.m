% 文件夹路径
folder_path = './originalImages/';  % 当前文件夹，如果图像在其他位置，修改为相应路径

% 获取文件夹中所有图片文件
img_files = dir(fullfile(folder_path, '*.png'));  % 假设图片格式为 PNG

% 总图片数量
num_images = numel(img_files);

% 分配比例
group_size = floor(num_images / 3);  % 每组的大小

% 目标尺寸
sizes = {[512, 512], [1024, 1024], [2048, 2048]};  % 目标分辨率

% 遍历所有图片并调整大小
for i = 1:num_images
    % 获取当前图像文件名
    img_name = img_files(i).name;
    
    % 读取图像
    img = imread(fullfile(folder_path, img_name));
    
    % 确定分配到哪个分辨率
    if i <= group_size
        target_size = sizes{1};  % 512x512
    elseif i <= 2 * group_size
        target_size = sizes{2};  % 1024x1024
    else
        target_size = sizes{3};  % 2048x2048
    end
    
    % 调整图像大小
    img_resized = imresize(img, target_size);
    
    % 显示原图和调整后的图像
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_resized), title(['Resized Image ', num2str(i)]);
    
    % 保存调整后的图像
    resized_img_name = sprintf('resized_%d_%dx%d.png', i, target_size(1), target_size(2));  % 保存为 resized_1_512x512.png 等
    imwrite(img_resized, fullfile(folder_path, resized_img_name));
end
