% close all; clear; clc;

mode = 2;
if mode == 1
    %  to jet256
    base_path = '/home/xjw/Projects/tmi2016_ks/predict/';
    list = dir([base_path '*.mat']);

    cmap = jet(256);
    for i = 1:length(list)
        load([base_path list(i).name]);
        img = (img - min(min(img))) / (max(max(img)) - min(min(img)));
        img = ind2rgb(uint8(img * 256), cmap);
        imwrite(img, [base_path 'jet256/' list(i).name(1:end - 4) '.png']);
    end
else
    % union original and the output
    new_out = img_test_NGH_1539946_01_8_16;
    for i = 1:500
        for j = 1:500
            ori = img_test_NGH_1539946_01_8_16(i, j, 1:3);
            out = test_img_test_NGH_1539946_01_8_16(i, j, 1:3);
            
            if sum(out(1:2)) < 30
                new_out(i, j, 1:3) = ori;
            else
                out(:, :, 3) = 0;
                new_out(i, j, 1:3) = 0.5 * ori + 0.5 * out;
            end
        end
    end
end