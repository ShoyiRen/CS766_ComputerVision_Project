function FeatureExtractMPIIGaze( )
%The main program doing the feature extraction

tic;
H = vision.TemplateMatcher;
load template.mat
for person = 0:14
    disp(['Processing Person ', num2str(person)]);
    %initialize the store construct for each person
    RightFeature = {};RightGaze = {};RightPose = {};
    LeftFeature = {};LeftGaze = {};LeftPose = {};
    NumImageDay = [];  
    
    %find the path where the data is stored
    PersonString = GeneratePerson(person);
    clear files
    % the files.mat store the filename in each day such as 'day01.mat' etc.
    filespath = ['./Normalized/', PersonString, '/files.mat'];
    load(filespath);
    
    for day = 1:size(files,1)
        %load the data of a specific date
        clear filenames data
        datapath = ['./Normalized/', PersonString, '/', files(day).name];
        load(datapath);
        
        %Right
        %get the original data from the dataset
        ImageData = data.right.image;
        ImageData = permute(ImageData,[2 3 1]);
        GazeData = data.right.gaze';
        PoseData = data.right.pose';
        
        %for each image, calculate the appearance feature
        for img = 1:size(GazeData,2)
            CurImage = ImageData(:,:,img);
            descriptor = Getfeatures(CurImage, H, RLeftCorner, RRightCorner, 'Right');
            RightFeature{day, img} = descriptor;
        end
        
        %extract the gaze and pose information
        RightGaze{day} = GazeData;
        RightPose{day} = PoseData;
        
        %Left
        %get the original data from the dataset
        ImageData = data.left.image;
        ImageData = permute(ImageData,[2 3 1]);
        GazeData = data.left.gaze';
        PoseData = data.left.pose'; 
        
        %for each image, calculate the appearance feature
        for img = 1:size(GazeData,2)
            CurImage = ImageData(:,:,img);
            descriptor = Getfeatures(CurImage, H, LLeftCorner, LRightCorner, 'Left');
            LeftFeature{day, img} = descriptor;
        end
        
        %extract the gaze and pose information
        LeftGaze{day} = GazeData;
        LeftPose{day} = PoseData;  
        
        %num of image in each day
        NumImageDay(day) = size(GazeData,2);
    end
    
    %save the result of each person to the given path
    savepath = ['./resize1/', PersonString, 'features.mat'];
    save(savepath, 'RightFeature', 'LeftFeature','RightGaze','LeftGaze',...
                    'RightPose', 'LeftPose','PersonString','NumImageDay');
    toc;
end
end

function PersonString = GeneratePerson(person)
%generate the string that matchs the filename given in the dataset
    if person < 10
        PersonString = ['p0', num2str(person)];
    else
        PersonString = ['p', num2str(person)];
    end
end


function descriptor = Getfeatures(img, H, LeftCorner, RightCorner, LR)
%this function doing the extraction of appearance features

    % flip the right eye images
    if(strcmp(LR,'Right'))
        img = fliplr(img);
    end
    
    %bilateral image filtering
    img = GPA(double(img), 40, 0.5, 1e-3, 'Gauss'); 
    img = uint8(img);
    
    %template matching
    Lloc = step(H, img, LeftCorner);
    Rloc = step(H, img, RightCorner);
    
    %align the image and crop it in a fixed aspect ratio
    if(~isempty(Lloc) && ~isempty(Rloc))
        Left = double(Lloc(1));
        Right = double(Rloc(1));
        Width = Right - Left;

        Up = max(0,double(Lloc(2)) - 2/5*Width);
        Down = min(size(img,1),double(Lloc(2)) + 1/5*Width);
        Height = Down - Up;
        
        %check whether the imcrop can be done without throwing an error
        if(Left < size(img,2)*(1/3) && Right > size(img, 2)*(2/3) && Up < Down)
            img = imcrop(img, [Left Up Width Height]);
        end
    end
    
    %the boundaries of the subregions
    [Height, Width] = size(img);
    h = round([1, 1/3*Height, 2/3*Height, Height]);
    w = round([1, 1/5*Width, 2/5*Width, 3/5*Width, 4/5*Width, Width]);
    
    %sum the intensity of subregions and normalize it
    descriptor = zeros(3,5);
    for i = 1:3
        for j = 1:5
            descriptor(i,j) = sum(sum(img(h(i):h(i+1),w(j):w(j+1))));        
        end
    end
    descriptor = double(reshape(descriptor', 15, []));
    descriptor = descriptor/norm(descriptor,1);
end








