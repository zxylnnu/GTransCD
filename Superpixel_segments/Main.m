clear; close all;

hsi=load('xxx.mat');hsi=hsi.river;
n_superpixels=[xxxx];

hsi=double(hsi);
[h,w,c]=size(hsi);
hsi=mapminmax( reshape(hsi,[h*w,c])');
hsi=reshape(hsi',[h,w,c]);
hsi = imfilter(hsi, fspecial('gaussian',[5,5]), 'replicate');
hsi=reshape(hsi,[h*w,c])';
pcacomps=pca(hsi);
I=pcacomps(:,[3,2,1])';
I=(mapminmax(I)+1)/2*255;
I=reshape(uint8(I)',[h,w,3]);
for i=1:3
    I(:,:,i)=imadjust(histeq(I(:,:,i))); 
end
I = imfilter(I, fspecial('unsharp',0.05), 'replicate');
E=uint8(zeros([h,w]));
sh=SuperpixelHierarchyMex(I,E,0.5); 
segmentmaps=zeros(size(n_superpixels,2),h,w);
for i=1:size(n_superpixels,2)
    GetSuperpixels(sh,n_superpixels(:,i));
    segmentmaps(i,:,:)=sh.label;
end

save xxx.mat segmentmaps