close all; clear; clc;

path = './predict/UNet_VGG16_WSI/mat/';
data_list = dir([path '*.mat']);

w = 102;
h = 71;
cmap = jet(256);
name = 'test_img_test_NGH-1539946-01';
for i = 0:w
    for j = 0:h
        try
            data = load([path name '_' num2str(i) '_' num2str(j) '.jpg.mat']);
            data = data.img;
            data = (data - min(min(data))) / (max(max(data)) - min(min(data)));
        catch
            data = single(zeros([500, 500]));
        end
%         data = ind2rgb(uint8(data * 256), cmap);
        
        if j == 0
            column = data;
        else
            column = cat(1, column, data);
        end
    end
    if i == 0
        map = column;
    else
        map = cat(2, map, column);
    end
end

% map_uint8 = ind2rgb(uint8(map * 256), cmap);
imwrite(uint8(map), './predict/UNet_VGG16_WSI/WSI_output/20x_original.tif');
save('./predict/UNet_VGG16_WSI/WSI_output/20x_original.mat', 'map');