function CrossValidation_LR( method, HP )
%cross validation work using LR
%method: either LeaveOneOut or TenFold
%HP: either 1 or 0

if nargin < 1
    method = 'LeaveOneOut';
    HP = 0;
end
load FeatureGroupCV.mat
Features = [];
Gazes = [];
Poses = [];
idx = [];

%group the chosen features of all people together
for person = 0:14
    idx = [idx,size(Features,2)];
    Features = [Features,TrainFeatures{person+1}];
    Gazes = [Gazes,TrainGazes{person+1}];
    Poses = [Poses,TrainPoses{person+1}];
end

%the index of each person in the grouped matrix
idx = [idx,size(Features,2)];

if(strcmp(method, 'LeaveOneOut'))
    switch HP
        case 0
            disp('Doing leave one out cross validation without headpose');
            [result,groundtruth] = LeaveOneOut(Features, Gazes,idx);
        case 1
            disp('Doing leave one out cross validation with headpose');
            [result,groundtruth] = LeaveOneOut([Features;Poses], Gazes,idx);
        otherwise
            disp('wrong input');
    end
elseif(strcmp(method, 'TenFold'))
	switch HP
        case 0
            disp('Doing 10-fold cross validation without headpose');
            [result,groundtruth] = TenFoldCV(Features, Gazes);
        case 1
            disp('Doing 10-fold cross validation with headpose');
            [result,groundtruth] = TenFoldCV([Features;Poses], Gazes);
        otherwise
            disp('wrong input');
    end    
else
    disp('wrong method'); 
    return;
end

%calculate the error
diff = cos(groundtruth(1,:)).*cos(result(1,:)).*cos(groundtruth(2,:) - result(2,:))...
        + sin(groundtruth(1,:)).*sin(result(1,:));
error = acos(diff)*180/pi;
disp(['mean: ', num2str(mean(error))]);
disp(['std: ', num2str(std(error))]);

filename = ['CVresult_',method,num2str(HP),'.mat'];
save( filename, 'result', 'groundtruth');
end

function [result,groundtruth] = LeaveOneOut(Features, Gazes, idx)
result = [];
groundtruth = [];
tic;
for person = 0:14
    disp(['Leaving peron ',num2str(person), ' out']);
    
    %separate training and test features
    TestFeature = Features(:,idx(person+1)+1:idx(person+2));
    TrainingFeature = Features;
    TrainingFeature(:,idx(person+1)+1:idx(person+2)) = [];
    
    %separate training and test gaze
    groundtruth = [groundtruth, Gazes(:,idx(person+1)+1:idx(person+2))];
    TrainingGaze = Gazes;
    TrainingGaze(:,idx(person+1)+1:idx(person+2)) = [];
    
    Training_number = size(TrainingFeature,2);

    A = TrainingFeature' \ TrainingGaze' ;
    Predict = TestFeature' * A;
    result = [result, Predict'];
    toc;
end
end

function [result,groundtruth] = TenFoldCV(Features, Gazes)
result = [];
groundtruth = [];

%randomize the features
idx = randperm(size(Features,2),size(Features,2));
Features = Features(:,idx);
Gazes = Gazes(:,idx);
N = int32(size(Features,2)/10);

tic;
for group = 0:9
    disp(['Leaving group ',num2str(group), ' out']);
    
    %get the start and end index of each group
    istart = group*N+1;
    if group == 9 
        iend = size(Features,2);
    else
        iend = (group+1)*N;
    end
    
    %separate training and test features
    TestFeature = Features(:,istart:iend);
    TrainingFeature = Features;
    TrainingFeature(:,istart:iend) = [];
    
    %separate training and test gazes   
    groundtruth = [groundtruth, Gazes(:,istart:iend)];
    TrainingGaze = Gazes;
    TrainingGaze(:,istart:iend) = [];
    
    Training_number = size(TrainingFeature,2);

    A = TrainingFeature' \ TrainingGaze' ;
    Predict = TestFeature' * A;
    result = [result, Predict'];
    toc;
end
end
