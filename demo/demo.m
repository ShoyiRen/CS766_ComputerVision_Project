disp('staring cross validation')
tic;
CrossValidation('LeaveOneOut',0);
toc;
tic;
CrossValidation('TenFold',0);
toc;
tic;
CrossValidation('LeaveOneOut',1);
toc;
tic;
CrossValidation('TenFold',1);
toc;