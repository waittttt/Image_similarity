% �ļ���·�������Ը�����Ҫ������
folder_path = './';  % ��ǰ�ļ��У����ͼ��������λ�ã��޸�Ϊ��Ӧ·��

% ���� image1.png �� image10.png
for i = 1:10
    % �����ļ���
    img_name = sprintf('image%d.png', i);
    
    % ��ȡͼ��
    img = imread(fullfile(folder_path, img_name));
    
    % ��Ӹ�˹����
    noisy_img = imnoise(img, 'gaussian', 0, 0.05);
    
    % ��ʾԭͼ�ͼ��������ͼ��
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(noisy_img), title(['Image with Gaussian Noise ', num2str(i)]);
    
    % ������������ͼ��
    noisy_img_name = sprintf('noisy_image%d.png', i);  % ����Ϊ noisy_image1.png ��
    imwrite(noisy_img, fullfile(folder_path, noisy_img_name));
end
