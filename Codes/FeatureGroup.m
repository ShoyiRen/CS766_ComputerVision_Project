function FeatureGroup(p)
%Randomly choose 1500x2 featrues for person p

TrainRightFeature = [];TrainRightGaze = [];TrainRightPose = [];
TrainLeftFeature = [];TrainLeftGaze = [];TrainLeftPose = [];

Directory = './resize1/';
person = p;
PersonString = GeneratePerson(person);
datapath = [Directory, PersonString, 'features.mat'];
load(datapath);

%groupd all features/gazes/poses of a person into a single matrix
for day = 1:size(NumImageDay,2)
    for img = 1:NumImageDay(day)
        %Right
        TrainRightFeature = [TrainRightFeature, RightFeature{day, img}];
            
        %Left
        TrainLeftFeature = [TrainLeftFeature, LeftFeature{day, img}];
    end
    TrainRightGaze = [TrainRightGaze, RightGaze{day}];
    TrainRightPose = [TrainRightPose, RightPose{day}];
    TrainLeftGaze = [TrainLeftGaze, LeftGaze{day}];
    TrainLeftPose = [TrainLeftPose, LeftPose{day}];
end

%randomly choose K features from each eye
N = size(TrainLeftFeature,2);
K = min(N,1500);
choiceL = randperm(N, K);
choiceR = randperm(N, K);
TrainFeature = [TrainLeftFeature(:,choiceL), TrainRightFeature(:,choiceR)];

%flip the horizontal axis or right gaze
%transform from 3D to 2D
TrainGaze = [TrainLeftGaze(:,choiceL),TrainRightGaze(:,choiceR)];
TrainGaze(1,K+1:end) = -TrainGaze(1,K+1:end);
theta = asin(-TrainGaze(2,:));
phi = atan2(-TrainGaze(1,:), -TrainGaze(3,:));
TrainGaze = [theta; phi];

%flip the horizontal axis or right pose
TrainPose = [TrainLeftPose(:,choiceL),TrainRightPose(:,choiceR)];
TrainPose(1,K+1:end) = -TrainPose(1,K+1:end);

save(['FeatureGroup', GeneratePerson(p),'.mat'], 'TrainFeature','TrainGaze','TrainPose');
end

function PersonString = GeneratePerson(person)
    if person < 10
        PersonString = ['p0', num2str(person)];
    else
        PersonString = ['p', num2str(person)];
    end
end
