function [TrainAcc,TestAcc,numSupVec,opt] = KFoldBetweenSubjFS(opt)
% ~~~~ K-Fold Classification ~~~~
%
% [TrainAcc,TestAcc,TrainR2,TestR2] = KFoldBetweenSubjFS(opt)
%
% opt - a struct containing input options
%
% Input Options:
%   opt.name - file name of text file
%   opt.outclass - 'binary','arousal', or 'valence' (default: binary)
%   opt.feat - array of features to include; check featureExtraction.m for
%              details (default: [5] i.e. Bicoh)
%   opt.subrem - portion of subjects to remove based on RMS values
%                (default: 0)
%   opt.vis - choose which of the two visits to test (default: [1,2])
%   opt.sov -
%       opt.sov.s - size of segmented data in seconds (default: 30)
%       opt.sov.ov - portion of overlap of segments (default: 0.5)
%   opt.weightAvg - 'on'/'off' uses weighted average instead of regular
%                   average to calculate accuracy (default: 'off')
%                   note: for binary classification only
%   opt.input - include data instead of loading it from function to save
%               time. Include as opt.input.Fear and opt.input.Happy
%   opt.save - 'on'/'off' choose whether or not to save data text file and
%              mat file (default = 'on') Note: must also include opt.name
%              if opt.save = 'on'
%   opt.k - value for k-fold cross-validation (default: 10)
%   opt.mRMR - which version of mRMR to use: 'MID' or 'MIQ' (default = MID)
%   opt.fCount - when selecting features, count by fCount. eg. fCount = 5,
%                featlist = [5,10,15,...] (default: 1)
%   opt.gamma - Kernel Scale input (default: auto)
%   opt.C - BoxConstraint input (if not specified, uses fitcsvm default:1)
% 
%
% Dependencies:
% featureExtraction.m - calcPSD.m frontalAlphaAsym chanCorr.m bispecEst.m
%   bicoherEst.m chanCPSD.m chanCoh.m
% loadFS.m
% outputClassAll.m
% removeSubjects.m
% 
% Note: must precalculate mRMR lists before running on wobbie

tic;
rng default;
clc;

%% Get inputs and set defaults
% NOT DEALING WITH VALENCE AND AROUSAL NOW
outclass = 'binary'; opt.outclass = outclass;
if isfield(opt,'subrem'), p = opt.subrem;
else, p = 0; opt.subrem = p; end
if isfield(opt,'feat'), feat = opt.feat;
else, feat = 5; opt.feat = feat; end
if isfield(opt,'sov'), S = opt.sov.s; OV = opt.sov.ov;
else, S = 30; OV = 0.5; opt.sov.s = S; opt.sov.ov = OV; end
if isfield(opt,'vis'), vislist = opt.vis;
else, vislist = [1,2]; opt.vis = vislist; end
if isfield(opt,'mRMR')
    if strcmp(opt.mRMR,'miq')||strcmp(opt.mRMR,'MIQ')
        miq = true;
    else
        miq = false;
    end
else
    miq = false; opt.mRMR = 'mid';
end
if isfield(opt,'fCount'), fCount = opt.fCount;
else, fCount = 1; opt.fCount = fCount; end
if isfield(opt,'weightAvg')
    if strcmp(opt.weightAvg,'on')
        wa = true;
    else
        wa = false;
    end
else
    wa = false; opt.weightAvg = 'off';
end
if isfield(opt,'k'), k = opt.k;
else, k = 10; opt.k = k; end
if isfield(opt,'save')
    if strcmp(opt.save,'off')
        sf = false;
    else
        if ~isfield(opt,'name')
            error('Must include name to save files');
        end
        name = opt.name;
        sf = true;
    end
else
    if ~isfield(opt,'name')
        error('Must include name to save files');
    end
    name = opt.name;
    sf = true; opt.save = 'on';
end
% if isfield(opt,'gamma')
%     g = true;
%     gamma = opt.gamma;
% else
%     g = false;
%     gamma = 'auto';
% end
g=false;
if isfield(opt,'C')
    c = true;
    Constraint = opt.C;
else
    c = false;
end
    

fprintf('\n--------------- K-Fold Classification ---------------\n\n');

%% Load the data and split into time windows
try
    if OV == 0
        filename = sprintf('FearHappy_S%d.mat',S);
        load(filename);
    elseif (OV>0)&&(OV<1)
        n = num2str(OV,2);
        filename = sprintf('FearHappy_S%d_OVp%s.mat',S,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Creating new segmented EEG data...\n\n');
    if OV==0
        [Fear, Happy] = dataSplitNonOV(S);
    elseif (OV>0)&&(OV<1)
        [Fear, Happy] = dataSplitOV(S,OV);
    end
end

%% Print out segment info
if OV>0
    fprintf('Segment selected: S%d OVp5\n\n',S);
else
    fprintf('Segment selected: S%d\n\n',S);
end

%% Print out mRMR method
if miq
    fprintf('mRMR method: MIQ\n\n');
else
    fprintf('mRMR method: MID\n\n');
end

%% Features for all data
fstr = '';
if sum(feat==1)>0, fstr = strcat(fstr, sprintf(' %s, ','PSD')); end
if sum(feat==2)>0, fstr = strcat(fstr, sprintf(' %s, ','FAA')); end
if sum(feat==3)>0, fstr = strcat(fstr, sprintf(' %s, ','ChanCorr')); end
if sum(feat==4)>0, fstr = strcat(fstr, sprintf(' %s, ','Bispec')); end
if sum(feat==5)>0, fstr = strcat(fstr, sprintf(' %s, ','Bicoh')); end
if sum(feat==6)>0, fstr = strcat(fstr, sprintf(' %s, ','Coh')); end
if sum(feat==7)>0, fstr = strcat(fstr, sprintf(' %s, ','CPSD')); end
if sum(feat==8)>0, fstr = strcat(fstr, sprintf(' %s, ','RMS')); end

fprintf('Features included:%s \n\n',fstr);
[FearFeatures, HappyFeatures] = featureExtraction(Fear,Happy,S,OV,feat);

%% Load Feature Selection list
% will load if available, otherwise returns zeros and fs=true

%#function cvpartition
[FeatSel,fs,CVO] = loadFSKfold(feat,miq,S,OV,k);
if fs
    error('No FS list found - Cannot Calculate mRMR on sharcnet');
%     fprintf('Calculating FeatSel using mRMR...\n\n'); %just to warn in advance
end

%% Cross Validation Partition
% k-fold cross validation
% CVO = cvpartition(sublist,'KFold',k);
fprintf('Performing %d-Fold Cross-Validation\n\n',k);


%% Output class for all data
[FearOutput, HappyOutput] = outputClassAll(Fear,Happy,outclass);
fprintf('Output class is %s.\n\n',outclass);


%% Prepare sublist and vislist
% Remove subjects with noisiest data (check removeSubjects.m for info)
% AND Remove subjects with invalid valence/arousal input
% (check preprocessOutputs.m for more info)
sublist = removeSubjects(Fear,Happy,p);
% if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%     sublist(sublist==78) = [];
%     sublist(sublist==91) = [];
%     sublist(sublist==99) = [];
% end
%%%%%%%%%%%%%%% JUST TRYING SHIT OUT
% % Trying to remove subjects that felt opposite emotion at least once (valence)
% % there's 47 of them...
% ivfvh = [3 11 13 14 15 20 23 24 26 30 38 39 40 42 43 45 49 51 52 53 56 58 60 63 65 71 75 76 77 78 80 81 82 88 90 91 92 96 97 98 100 101 104 105 106 107 110];
% for i = 1:length(ivfvh)
%     sublist(sublist==ivfvh(i)) = [];
% end

%% Store accuracy

% Store Feature Selection array
featlist = fCount:fCount:size(FearFeatures{1,1,1},2);
totalfeats = length(featlist);

TrainAcc = zeros(CVO.NumTestSets,length(vislist),totalfeats);
TestAcc = zeros(CVO.NumTestSets,length(vislist),totalfeats);
% TrainR2 = zeros(CVO.NumTestSets,length(vislist),totalfeats);
% TestR2 = zeros(CVO.NumTestSets,length(vislist),totalfeats);

TestScore = cell(CVO.NumTestSets,length(vislist),totalfeats);

numSupVec = zeros(CVO.NumTestSets,length(vislist),totalfeats);
numObs = zeros(CVO.NumTestSets,length(vislist),totalfeats);

opt.sub = sublist;
opt.featlist = featlist;

%% Learn
for nVisit = vislist
    parfor s = 1:CVO.NumTestSets
        nSubs = sublist(CVO.test{s});
        tSubs = sublist(CVO.training{s});
        
        %% Prepare Training matrix
        TrainFeatures = [];
        TrainOutput = [];
        TrainFPred = cell(length(tSubs),8); % F1,F2,F3,F4,H1,H2,H3,H4
        TrainOPred = cell(length(tSubs),8);
        for t = 1:length(tSubs)
            for n = 1:4
                TrainFeatures = cat(1,TrainFeatures,...
                    cell2mat(FearFeatures(tSubs(t),nVisit,n)),...
                    cell2mat(HappyFeatures(tSubs(t),nVisit,n)));
                TrainOutput = cat(1,TrainOutput,...
                    cell2mat(FearOutput(tSubs(t),nVisit,n)),...
                    cell2mat(HappyOutput(tSubs(t),nVisit,n)));
                TrainFPred{t,n} = cell2mat(FearFeatures(tSubs(t),nVisit,n));
                TrainFPred{t,n+4} = cell2mat(HappyFeatures(tSubs(t),nVisit,n));
                TrainOPred{t,n} = cell2mat(FearOutput(tSubs(t),nVisit,n));
                TrainOPred{t,n+4} = cell2mat(HappyOutput(tSubs(t),nVisit,n));
            end
        end
        
        % Shuffle -- is this necessary?
        ind = randperm(size(TrainOutput,1));
        TrainFeatures = TrainFeatures(ind,:);
        TrainOutput = TrainOutput(ind,:);
        
        %% Prepare Test Matrix
%         TestFeatures = [];
%         TestOutput = [];
        TestFPred = cell(length(nSubs),8); % F1,F2,F3,F4,H1,H2,H3,H4
        TestOPred = cell(length(nSubs),8);
        for t = 1:length(nSubs)
            for n = 1:4
%                 TestFeatures = cat(1,TestFeatures,...
%                     cell2mat(FearFeatures(nSubs(t),nVisit,n)),...
%                     cell2mat(HappyFeatures(nSubs(t),nVisit,n)));
%                 TestOutput = cat(1,TestOutput,...
%                     cell2mat(FearOutput(nSubs(t),nVisit,n)),...
%                     cell2mat(HappyOutput(nSubs(t),nVisit,n)));
                TestFPred{t,n} = cell2mat(FearFeatures(nSubs(t),nVisit,n));
                TestFPred{t,n+4} = cell2mat(HappyFeatures(nSubs(t),nVisit,n));
                TestOPred{t,n} = cell2mat(FearOutput(nSubs(t),nVisit,n));
                TestOPred{t,n+4} = cell2mat(HappyOutput(nSubs(t),nVisit,n));
            end
        end
        
        %% Feature Selection on Training Matrix
%         if fs
%             TrainFeatS = bsxfun(@minus,TrainFeatures,mean(TrainFeatures));
%             TrainFeatS = bsxfun(@rdivide,TrainFeatS,std(TrainFeatures));
%             if miq
%                 FeatSel(s,:,nVisit) = mrmr_miq_d(TrainFeatS,TrainOutput,totalfeats);
%             else
%                 FeatSel(s,:,nVisit) = mrmr_mid_d(TrainFeatS,TrainOutput,totalfeats);
%             end
%         end
        
        %% create temp arrays for parfor
%         tempTrainAcc = zeros(totalfeats,1);
%         tempTestAcc = zeros(totalfeats,1);
%         tempnumSupVec = zeros(totalfeats,1);
%         tempnumObs = zeros(totalfeats,1);
        
        %% Feature Selection loop
        for f = 1:totalfeats
            TrainFPredFS = cell(size(TrainFPred));
            TestFPredFS = cell(size(TestFPred));
            
            % Select mRMR features from feature sets
            TrainFeaturesFS = TrainFeatures(:,FeatSel(s,1:featlist(f),nVisit));
%             TestFeaturesFS = TestFeatures(:,FeatSel(s,1:f,nVisit));
            for t = 1:length(nSubs)
                for n = 1:8
                    TestFPredFS{t,n} = TestFPred{t,n}(:,FeatSel(s,1:featlist(f),nVisit));
                end
            end
            for t = 1:length(tSubs)
                for n = 1:8
                    TrainFPredFS{t,n} = TrainFPred{t,n}(:,FeatSel(s,1:featlist(f),nVisit));
                end
            end
            
            % Train and Test
%             if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%                 % Measures MSE as Accuracy
%                 model = fitrsvm(TrainFeaturesFS,TrainOutput,'Standardize',true,...
%                     'KernelFunction','rbf','KernelScale','auto');
%                 
%                 % Training accuracy
%                 acc = zeros(size(TrainOPred));
%                 T = cell(size(TrainOPred));
%                 for t = 1:length(tSubs)
%                     for v = 1:8
%                         T{t,v} = predict(model,TrainFPredFS{t,v});
%                         acc(t,v) = mean((T{t,v}-TrainOPred{t,v}).^2);
%                     end
%                 end
%                 TrainAcc(s,nVisit,f) = mean(mean(acc,2));
%                 Tr = predict(model,TrainFeaturesFS);
%                 TrainR2(s,nVisit,f) = corr2(TrainOutput,Tr)^2;
%                 
%                 % Test accuracy
%                 acc = zeros(size(TestOPred));
%                 C = cell(size(TestOPred));
%                 for t = 1:length(nSubs)
%                     for v = 1:8
%                         C{t,v} = predict(model,TestFPredFS{t,v});
%                         acc(t,v) = mean((C{t,v}-TestOPred{t,v}).^2);
%                     end
%                 end
%                 TestAcc(s,nVisit,f) = mean(mean(acc,2));
%                 Ts = predict(model,TestFeaturesFS);
%                 TestR2(s,nVisit,f) = corr2(TestOutput,Ts)^2;
%                 
%                 fprintf('>> Number of features selected: %d\nTesting on K-Fold %d\t Visit %d .....\n\tTrain:\tMSE = %3.2f \tR2 = %1.6f\n\tTest:\tMSE = %3.2f \tR2 = %1.6f\n',f,s,nVisit,TrainAcc(s,nVisit,f),TrainR2(s,nVisit,f),TestAcc(s,nVisit,f),TestR2(s,nVisit,f));
%             else    % binary
                if g
                    if c
                        model = fitcsvm(TrainFeaturesFS,TrainOutput,'Standardize',true,...
                            'KernelFunction','rbf','KernelScale',gamma,'ScoreTransform','logit',...
                            'BoxConstraint',Constraint);
                    else
                        model = fitcsvm(TrainFeaturesFS,TrainOutput,'Standardize',true,...
                            'KernelFunction','rbf','KernelScale',gamma,'ScoreTransform','logit');
                    end
                else
                    if c
                        model = fitcsvm(TrainFeaturesFS,TrainOutput,'Standardize',true,...
                            'KernelFunction','rbf','KernelScale','auto','ScoreTransform','logit',...
                            'BoxConstraint',Constraint);
                    else
                        model = fitcsvm(TrainFeaturesFS,TrainOutput,'Standardize',true,...
                            'KernelFunction','rbf','KernelScale','auto','ScoreTransform','logit');
                    end
                end
                numSupVec(s,nVisit,f) = sum(model.IsSupportVector);
                numObs(s,nVisit,f) = model.NumObservations;

                % Training Accuracy
                acc = zeros(size(TrainOPred));
                for t = 1:length(tSubs)
                    for v = 1:8
                        [T,Tscore] = predict(model,TrainFPredFS{t,v});
                        if wa
                            % have not tested yet ****
                            m = mean(Tscore(:,2))>=0.5;
                            acc(t,v) = mean(m==TrainOPred{t,v})>=0.5;
                        else
                            acc(t,v) = mean(T==TrainOPred{t,v})>=0.5;
                        end
                    end
                end
                TrainAcc(s,nVisit,f) = mean(mean(acc,2));
                
                % Test Accuracy
                acc = zeros(size(TestOPred));
                TestScore{s,nVisit,f} = cell(length(nSubs),8,3); % T, Tscore, TestOPred
                for t = 1:length(nSubs)
                    for v = 1:8
                        [C,Cscore] = predict(model,TestFPredFS{t,v});
                        TestScore{s,nVisit,f}{t,v,1} = C;
                        TestScore{s,nVisit,f}{t,v,2} = Cscore;
                        TestScore{s,nVisit,f}{t,v,3} = TestOPred{t,v};
                        if wa
                            m = mean(Cscore(:,2))>=0.5;
                            acc(t,v) = mean(m==TestOPred{t,v})>=0.5;
                        else
                            acc(t,v) = mean(C==TestOPred{t,v})>=0.5;
                        end
                    end
                end
                TestAcc(s,nVisit,f) = mean(mean(acc,2));
                
                fprintf('>> Number of features selected: %d\nTesting on K-Fold %d\t Visit %d .....\n\tTrain:\tAcc = %3.2f\n\tTest:\tAcc = %3.2f\n',featlist(f),s,nVisit,100*TrainAcc(s,nVisit,f),100*TestAcc(s,nVisit,f));
%             end
        end
        %% Add accuracies to TestAcc and TrainAcc
%          = tempTrainAcc;
%          = tempTestAcc;
%          = tempnumSupVec;
%          = tempnumObs;
        
    end
end

%% JUST FOR SAVING INTO TEXT FILE - must be outside of parallel loop
% if sf
%     for nVisit = vislist
%         for s = 1:CVO.NumTestSets
%             for f = 1:totalfeats
%                 if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%                     fprintf(fileID,'>> Number of features selected: %d\nTesting on K-Fold %d\t Visit %d .....\n\tTrain:\tMSE = %3.2f \tR2 = %1.6f\n\tTest:\tMSE = %3.2f \tR2 = %1.6f\n',f,s,nVisit,TrainAcc(s,nVisit,f),TrainR2(s,nVisit,f),TestAcc(s,nVisit,f),TestR2(s,nVisit,f));
%                 else
%                     fprintf(fileID,'>> Number of features selected: %d\nTesting on K-Fold %d\t Visit %d .....\n\tTrain:\tAcc = %3.2f\n\tTest:\tAcc = %3.2f\n',f,s,nVisit,100*TrainAcc(s,nVisit,f),100*TestAcc(s,nVisit,f));
%                 end
%             end
%         end
%     end
% end

%% Save output
if sf
    if exist('Results13','dir')~=7
        !mkdir Results13
    end
    if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
        fnamemat = ['Results13/',name,'.mat'];
        save(fnamemat,'TrainAcc','TrainR2','TestAcc','TestR2','opt','-v7.3');
        toc
    else
        % Check numObs
        if length(unique(numObs))==1
            numObs = unique(numObs);
            opt.numObs = numObs;
        else
            opt.numObs = numObs;
        end
        fnamemat = ['Results13/',name,'.mat'];
        save(fnamemat,'TrainAcc','TestAcc','numSupVec','opt','TestScore','-v7.3');
        toc
    end
else
    toc
end


%% Print out the average accuracy
if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%     meanTrainAcc = mean(TrainAcc);
%     stdTrainAcc = std(TrainAcc);
%     meanTrainR2 = mean(TrainR2);
%     stdTrainR2 = std(TrainR2);
%     meanTestAcc = mean(TestAcc);
%     stdTestAcc = std(TestAcc);
%     meanTestR2 = mean(TestR2);
%     stdTestR2 = std(TestR2);
%     if sf
%         fprintf(fileID,'\n\n-----\nVisit 1:\n\tMean Training MSE = %3.2f \t Stdev = %3.2f \n\tMean Training R2 = %1.6f \t Stdev = %1.6f \n',meanTrainAcc(1),stdTrainAcc(1),meanTrainR2(1),stdTrainR2(1));
%         fprintf(fileID,'\tMean Testing  MSE = %3.2f \t Stdev = %3.2f \n\tMean Testing  R2 = %1.6f \t Stdev = %1.6f \n',meanTestAcc(1),stdTestAcc(1),meanTestR2(1),stdTestR2(1));
%         fprintf(fileID,'\n\n-----\nVisit 2:\n\tMean Training MSE = %3.2f \t Stdev = %3.2f \n\tMean Training R2 = %1.6f \t Stdev = %1.6f \n',meanTrainAcc(2),stdTrainAcc(2),meanTrainR2(2),stdTrainR2(2));
%         fprintf(fileID,'\tMean Testing  MSE = %3.2f \t Stdev = %3.2f \n\tMean Testing  R2 = %1.6f \t Stdev = %1.6f \n',meanTestAcc(2),stdTestAcc(2),meanTestR2(2),stdTestR2(2));
%     end
%     fprintf('\n\n-----\nVisit 1:\n\tMean Training MSE = %3.2f \t Stdev = %3.2f \n\tMean Training R2 = %1.6f \t Stdev = %1.6f \n',meanTrainAcc(1),stdTrainAcc(1),meanTrainR2(1),stdTrainR2(1));
%     fprintf('\tMean Testing  MSE = %3.2f \t Stdev = %3.2f \n\tMean Testing  R2 = %1.6f \t Stdev = %1.6f \n',meanTestAcc(1),stdTestAcc(1),meanTestR2(1),stdTestR2(1));
%     fprintf('\n\n-----\nVisit 2:\n\tMean Training MSE = %3.2f \t Stdev = %3.2f \n\tMean Training R2 = %1.6f \t Stdev = %1.6f \n',meanTrainAcc(2),stdTrainAcc(2),meanTrainR2(2),stdTrainR2(2));
%     fprintf('\tMean Testing  MSE = %3.2f \t Stdev = %3.2f \n\tMean Testing  R2 = %1.6f \t Stdev = %1.6f \n',meanTestAcc(2),stdTestAcc(2),meanTestR2(2),stdTestR2(2));
else    % binary accuracy
%     meanTrainAcc = mean(TrainAcc);
%     stdTrainAcc = std(TrainAcc);
    meanTestAcc = mean(TestAcc);
%     stdTestAcc = std(TestAcc);
    [TestMax,nFeatMax] = max(meanTestAcc,[],3);
    fprintf('\n\n-----\nVisit 1:\n\tMax Test Acc = %3.2f \t nFeats = %3.2f \n',100*TestMax(1),nFeatMax(1)*fCount);
    fprintf('\n\n-----\nVisit 2:\n\tMax Test Acc = %3.2f \t nFeats = %3.2f \n',100*TestMax(2),nFeatMax(2)*fCount);
end

end
