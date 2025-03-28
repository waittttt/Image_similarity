% �ļ���·��
folder_path = './originalImages/';  % ��ǰ�ļ��У����ͼ��������λ�ã��޸�Ϊ��Ӧ·��

% ��ȡ�ļ���������ͼƬ�ļ�
img_files = dir(fullfile(folder_path, '*.png'));  % ����ͼƬ��ʽΪ PNG

% ��ͼƬ����
num_images = numel(img_files);

% �������
group_size = floor(num_images / 3);  % ÿ��Ĵ�С

% Ŀ��ߴ�
sizes = {[512, 512], [1024, 1024], [2048, 2048]};  % Ŀ��ֱ���

% ��������ͼƬ��������С
for i = 1:num_images
    % ��ȡ��ǰͼ���ļ���
    img_name = img_files(i).name;
    
    % ��ȡͼ��
    img = imread(fullfile(folder_path, img_name));
    
    % ȷ�����䵽�ĸ��ֱ���
    if i <= group_size
        target_size = sizes{1};  % 512x512
    elseif i <= 2 * group_size
        target_size = sizes{2};  % 1024x1024
    else
        target_size = sizes{3};  % 2048x2048
    end
    
    % ����ͼ���С
    img_resized = imresize(img, target_size);
    
    % ��ʾԭͼ�͵������ͼ��
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_resized), title(['Resized Image ', num2str(i)]);
    
    % ����������ͼ��
    resized_img_name = sprintf('resized_%d_%dx%d.png', i, target_size(1), target_size(2));  % ����Ϊ resized_1_512x512.png ��
    imwrite(img_resized, fullfile(folder_path, resized_img_name));
end
