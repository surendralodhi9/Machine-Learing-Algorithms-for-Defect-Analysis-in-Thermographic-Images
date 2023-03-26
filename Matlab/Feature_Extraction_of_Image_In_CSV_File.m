
srcDir = 'C:\Users\Administrator\Desktop\Mini Project\ML\temp';
desDir = 'C:\Users\Administrator\Desktop\Mini Project\IR Images\result\Data set files';

filePattern = fullfile(srcDir, '*.jpg');
files = dir(filePattern);

for k = 1:length(files)
  baseFileName = files(k).name;
  fullFileName = fullfile(srcDir, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  
  
  originalRGBImage = imread(fullFileName);
  grayImage = min(originalRGBImage, [], 3); % Useful for finding image and color map regions of image.
  % Get the dimensions of the image.  numberOfColorChannels should be = 3.
  [rows, columns, numberOfColorChannels] = size(originalRGBImage);
  % Crop off the surrounding clutter to get the RGB image.
  rgbImage = imcrop(originalRGBImage, [0, 0, columns, rows]);

  % Get the dimensions of the image.  
  % numberOfColorBands should be = 3.
  [rows, columns, numberOfColorChannels] = size(rgbImage);
  % Convert from an RGB image to a grayscale, indexed, thermal image.
  indexedImage = rgb2ind(rgbImage,32);
  % Display the thermal image.
  highTemp =0;% 28.9;
  % Define the temperature at the dark end of the scale
  % This will probably be the low temperature.
  lowTemp = 2;%0.9;
  % Scale the image so that it's actual temperatures
  thermalImage = lowTemp + (highTemp - lowTemp) * mat2gray(indexedImage);
  intensityImage = rgb2gray(rgbImage);
  
  d=10;%number of defect
  R=[103,62,19,18,61,99,100,60,65,99];
  C=[179,184,185,144,14,134,94,97,57,55];
  depth=[0,0,0,0,0,0,0,0,1,1];
  for i=1:d
      temperature=thermalImage(R(i),C(i));%7
      index=indexedImage(R(i),C(i));%5
      intensity=intensityImage(R(i),C(i));%6
      gray=grayImage(R(i),C(i));%4
      rgbColor = impixel(rgbImage, C(i), R(i));%(column,row)
      red=rgbColor(1);%1
      green=rgbColor(2);%2
      blue=rgbColor(3);%3

      %Making list as single row
  
      row = randn(1,6);
      row(1,1)=red;
      row(1,2)=green;
      row(1,3)=blue;
      row(1,4)=gray;
      row(1,5)=index;
      row(1,6)=intensity;
      %row(1,7)=temperature;
      %row(1,8)=0.5;%depth of defect
      %Writing data
      fileName= fullfile(desDir, 'input1.csv');
      dlmwrite(fileName,row,'delimiter',',','-append');
  
      %output file of depth
      output = randn(1,1);
      output(1,1)=temperature;
      fileName= fullfile(desDir, 'output1.csv');
      dlmwrite(fileName,output,'delimiter',',','-append');
      
      
      
      
  end
      
  
  
end

