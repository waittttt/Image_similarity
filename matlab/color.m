% �ļ���·�������Ը�����Ҫ������
folder_path = './';  % ��ǰ�ļ��У����ͼ��������λ�ã��޸�Ϊ��Ӧ·��

% ���� image1.png �� image10.png
for i = 1:10
    % �����ļ���
    img_name = sprintf('image%d.png', i);
    
    % ��ȡͼ��
    img = imread(fullfile(folder_path, img_name));
    
    % ��ͼ��ת��Ϊdouble�����Ա㴦��
    img_double = im2double(img);
    
    % ͳһ�������ȺͶԱȶ�
    % �������ǽ�ͼ���������20%������ǿ�Աȶ�
    img_adjusted = img_double * 1.2;  % ��������
    img_adjusted = imadjust(img_adjusted, [0.2 0.8], [0 1]);  % ��ǿ�Աȶ�
    
    % ���������ͼ��ת����uint8����
    img_adjusted = im2uint8(img_adjusted);
    
    % ��ʾԭͼ�͵������ͼ��
    subplot(1, 2, 1), imshow(img), title(['Original Image ', num2str(i)]);
    subplot(1, 2, 2), imshow(img_adjusted), title(['Adjusted Image ', num2str(i)]);
    
    % ����������ͼ��
    adjusted_img_name = sprintf('adjusted_image%d.png', i);  % ����Ϊ adjusted_image1.png ��
    imwrite(img_adjusted, fullfile(folder_path, adjusted_img_name));
end
