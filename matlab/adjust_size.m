% �ļ���·�������Ը�����Ҫ������
folder_path = './';  % ��ǰ�ļ��У����ͼ��������λ�ã��޸�Ϊ��Ӧ·��

% ����ͳһ��Ŀ���С������ 512x512��
target_size = [1024, 1024];  % Ŀ���С���к��зֱ�Ϊ�߶ȺͿ��

% ���� image1.png �� image10.png
for i = 1:10
    % �����ļ���
    img_name = sprintf('image%d.png', i);
    
    % ��ȡͼ��
    img = imread(fullfile(folder_path, img_name));
    
    % ����ͼ���С
    img_resized = imresize(img, target_size);  % ��ͼ�����ΪĿ���С
    
    % ��ʾԭͼ�͵������ͼ��
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_resized), title(['Resized Image ', num2str(i)]);
    
    % ����������ͼ��
    resized_img_name = sprintf('resized_image%d.png', i);  % ����Ϊ resized_image1.png ��
    imwrite(img_resized, fullfile(folder_path, resized_img_name));
end
