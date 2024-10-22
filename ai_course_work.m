clc
clear All
clear

%reading the dataset
digitDatasetPath = fullfile("archive/dataset/");
imds = imageDatastore(digitDatasetPath, ...
'IncludeSubfolders', true, ...
'LabelSource', 'foldernames', ...
'FileExtensions', '.jpeg', ... % Change the extension if needed
'ReadFcn', @(filename) preprocessImage(filename));

trainDatasetSize = 20;

%exploring the dataset
figure;
perm = randperm(100,20); 
for i = 1:trainDatasetSize
    subplot(4,5,i); 
    imshow(imds.Files{perm(i)});
end

%preprocessing steps
labelCount = countEachLabel(imds);


 
%Step 3: Traditional IM Techniques
%Filtering and convolution (e.g blurring)
%Edge Detection
%Image Segmentation
%THRESHHOLDING
%MORPHOLOGICAL OPERATIONS
%Image Enhancement
%Image Restoration
%Feature Detection and Decription 
%Noise Removal 
%Batch Processing
%Background Subtraction
%Erosion and Dilation
%Image Filtering


figure;
perm = randperm(100,20); 
for i = 1:trainDatasetSize

    img = imread(imds.Files{perm(i)});

    % Convert the image to grayscale
    gs = rgb2gray(img);
    
    % Normalize the pixel values
    gs = im2double(gs) / 255;

     %Image Enhancement
    gsAdj = imadjust(gs);%Improve the contrast of images(imadjust)
    subplot(4,5,i); 

   
    %Filtering and convolution
    kernel = fspecial('Gaussian', 3, 3); % Create a Gaussian kernel

    %Apply the Gaussian filter to the image
    filtered_image = imfilter(img, kernel);

    imshow(imds.Files{perm(i)});
    imshowpair(filtered_image,img,"montage")
    
end


%Step 4: Feature Extraction
%Histrogram of Orieintation (HOG)
%Scale Invariant Features Transform(SIFT)
%Local Binary Patterns(LBP)
%Color Histrogram
%Gabor Filter
%Principal Component Analysis (PCA)


% loop starts.......
figure;
for i = 1:trainDatasetSize

    im = imread(imds.Files{perm(i)});
    %Histrogram
    imhist(im);
    
    %Histrogram of Orieintation (HOG)
    [featureVector,hogVisualization] = extractHOGFeatures(im,'CellSize',[32,32]);
    subplot(4,5,i); 
    imshow(im); 
    hold on;
    plot(hogVisualization);

end

figure;
for i = 1:trainDatasetSize

    im = imread(imds.Files{perm(i)});

    subplot(4,5,i); 
 
    %Hough Transform
    BW = edge(gs,'canny');
    
    [H,T,R] = hough(BW,'RhoResolution',0.5,'Theta',-90:0.5:89);
    
    subplot(4,5,i);
    %imshow(RGB);
    title('Animal');
   
    imshow(imadjust(rescale(H)),'XData',T,'YData',R,'InitialMagnification','fit');
    title('Hough transform of Animal.jpeg');
    xlabel('\theta'),
    ylabel('\rho');
    axis on, 
    axis normal, 
    hold on;
    colormap(gca,hot);
    
    
    imshow(imadjust(rescale(H)),'XData',T,'YData',R,'InitialMagnification','fit');
    title('Hough Animal-10');
    xlabel('\theta')
    ylabel('\rho');
    axis on, axis normal;
    colormap(gca,hot)

end



%Spliting the data into training and testing (80% training and 20% testing data)

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomize');

%Convolutional neural network architecture. 
layers = [
imageInputLayer([256 256 3]) % size of image
convolution2dLayer(3,64,'Padding','same') % 3x3 without using any padding with 64 filters
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2) %Max pooling layer (2x2, stride=2)
convolution2dLayer(3,128,'Padding','same') % layer 2 with 128 filters
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2) 
convolution2dLayer(3,256,'Padding','same') %layer 3 with 256 filters
batchNormalizationLayer %	Batch normalization layer
reluLayer % ReLU activation layer
fullyConnectedLayer(10) %10 outputs bcz 0-9 total 10 digits are
softmaxLayer  % Softmax activation layer
classificationLayer]; % Classification layer

%Specify Training Options
options = trainingOptions('sgdm', ...
'InitialLearnRate',0.001, ...   %Initial learning rate: 0.001
'MaxEpochs',30, ...             %Maximum number of epochs: 30
'Shuffle','every-epoch', ...    %Shuffle: every epoch
'ValidationData',imdsValidation, ... 
'ValidationFrequency',30, ...   %Validation frequency: 30 epochs
'Verbose',false, ...            %Verbose: false
'Plots','training-progress');   %Plots: training-progress



% Train Network Using Training Data

net = trainNetwork(imdsTrain,layers,options);

%Classify Validation Images and Compute Accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);




% Define resize images function
function img = preprocessImage(filename)

    % Read and preprocess the image

    originalImage = imread(filename);

 

    % Resize the image to 256x256 pixels (adjust if needed)

    img = imresize(originalImage, [256, 256]);

 

    % Ensure that the image has 3 channels (RGB)

    if size(img, 3) ~= 3
    
        % Handle grayscale images by converting them to RGB
    
        img = cat(3, img, img, img);
    
    end

end
