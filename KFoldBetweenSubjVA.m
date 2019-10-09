function [TrainAcc,TestAcc] = KFoldBetweenSubjVA(opt)
% ~~~~ K-Fold Classification ~~~~
%
% [TrainAcc,TestAcc,TrainR2,TestR2] = KFoldBetweenSubj(opt)
%
% opt - a struct containing input options
%
% Input Options:
%   opt.name - file name of text file
%   opt.outclass - 'binary','arousal', or 'valence' (default: binary)
%   opt.feat - array of features to include; check featureExtraction.m for
%              details (default: [1,2,3,5] i.e. PSD,FAA,ChanCorr,Bicoh)
%   opt.subrem - portion of subjects to remove based on RMS values
%                (default: 0)
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
% 


% Do feature selection using mRMR
% Try different feature sets separately and together
% Try chosen important time frames for each video
% Rewrite to automatically write output to file
% (also print which person is being calculated in HOSA features)
tic;
rng default;
clc;


%% Get inputs and set defaults
if isfield(opt,'outclass'), outclass = opt.outclass;
else, outclass = 'binary'; end
if isfield(opt,'subrem'), p = opt.subrem;
else, p = 0; end
if isfield(opt,'feat'), feat = opt.feat;
else, feat = [1,2,3,5]; end
if isfield(opt,'sov'), s = opt.sov.s; ov = opt.sov.ov;
else, s = 30; ov = 0.5; end
if isfield(opt,'weightAvg')
    if strcmp(opt.weightAvg,'on')
        wa = true;
    else
        wa = false;
    end
else
    wa = false;
end
if isfield(opt,'k'), k = opt.k;
else, k = 10; end
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
    sf = true;
end

%% Name and open file
if sf
    fname = [pwd,'/Results12/'];
    fnametxt = [fname,name,'.txt'];
    fileID = fopen(fnametxt,'w');
end

if sf, fprintf(fileID,'\n--------------- K-Fold Classification ---------------\n\n'); end
fprintf('\n--------------- K-Fold Classification ---------------\n\n');


%% Load the data and split into time windows
if isfield(opt,'input')
    Fear = opt.input.Fear;
    Happy = opt.input.Happy;
else
    try
        if ov == 0
            filename = sprintf('/home/sara/Documents/Lallemande/FearHappy_S%d.mat',s);
            load(filename);
        elseif (ov>0)&&(ov<1)
            n = num2str(ov,2);
            filename = sprintf('/home/sara/Documents/Lallemande/FearHappy_S%d_OVp%s.mat',s,n(3:end));
            load(filename);
        else
            error('Invalid input for OV');
        end
    catch
        if sf, fprintf(fileID,'Creating new segmented EEG data...\n\n'); end
        fprintf('Creating new segmented EEG data...\n\n');
        if ov==0
            [Fear, Happy] = dataSplitNonOV(s);
        elseif (ov>0)&&(ov<1)
            [Fear, Happy] = dataSplitOV(s,ov);
        end
    end
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

if sf, fprintf(fileID,'Features included:%s \n\n',fstr); end
fprintf('Features included:%s \n\n',fstr);
[FearFeatures, HappyFeatures] = featureExtraction(Fear,Happy,s,ov,feat);

%% Output class for all data
[FearOutput, HappyOutput] = outputClassNew(Fear,Happy,outclass);
if sf, fprintf(fileID,'Output class is %s.\n\n',outclass); end
fprintf('Output class is %s.\n\n',outclass);

%% Prepare sublist and vislist
% Remove subjects with noisiest data (check removeSubjects.m for info)
% AND Remove subjects with invalid valence/arousal input
% (check preprocessOutputs.m for more info)
sublist = removeSubjects(Fear,Happy,p);
if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
    sublist(sublist==78) = [];
    sublist(sublist==91) = [];
    sublist(sublist==99) = [];
end
%%%%%%%%%%%%%%% JUST TRYING SHIT OUT
% % Trying to remove subjects that felt opposite emotion at least once (valence)
% % there's 47 of them...
% ivfvh = [3 11 13 14 15 20 23 24 26 30 38 39 40 42 43 45 49 51 52 53 56 58 60 63 65 71 75 76 77 78 80 81 82 88 90 91 92 96 97 98 100 101 104 105 106 107 110];
% for i = 1:length(ivfvh)
%     sublist(sublist==ivfvh(i)) = [];
% end

% Choose from visit 1 and/or visit 2
vislist = [1,2];


%% Cross Validation Partition
% k-fold cross validation
CVO = cvpartition(sublist,'KFold',k);
if sf, fprintf(fileID,'Performing %d-Fold Cross-Validation\n\n',k); end
fprintf('Performing %d-Fold Cross-Validation\n\n',k);

%% Store accuracy
TrainAcc = zeros(CVO.NumTestSets,length(vislist));
TestAcc = zeros(CVO.NumTestSets,length(vislist));
TrainR2 = zeros(CVO.NumTestSets,length(vislist));
TestR2 = zeros(CVO.NumTestSets,length(vislist));

%% Learn
for nVisit = vislist
    parfor s = 1:CVO.NumTestSets
        nSubs = sublist(CVO.test(s));
        tSubs = sublist(CVO.training(s));
        
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
        TestFeatures = [];
        TestOutput = [];
        TestFPred = cell(length(nSubs),8); % F1,F2,F3,F4,H1,H2,H3,H4
        TestOPred = cell(length(nSubs),8);
        for t = 1:length(nSubs)
            for n = 1:4
                TestFeatures = cat(1,TestFeatures,...
                    cell2mat(FearFeatures(nSubs(t),nVisit,n)),...
                    cell2mat(HappyFeatures(nSubs(t),nVisit,n)));
                TestOutput = cat(1,TestOutput,...
                    cell2mat(FearOutput(nSubs(t),nVisit,n)),...
                    cell2mat(HappyOutput(nSubs(t),nVisit,n)));
                TestFPred{t,n} = cell2mat(FearFeatures(nSubs(t),nVisit,n));
                TestFPred{t,n+4} = cell2mat(HappyFeatures(nSubs(t),nVisit,n));
                TestOPred{t,n} = cell2mat(FearOutput(nSubs(t),nVisit,n));
                TestOPred{t,n+4} = cell2mat(HappyOutput(nSubs(t),nVisit,n));
            end
        end
        
        %% Train and Test
        if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%             % Measures MSE as Accuracy
%             model = fitrsvm(TrainFeatures,TrainOutput,'Standardize',true,...
%                 'KernelFunction','rbf','KernelScale','auto');
%             
%             % Training accuracy
%             acc = zeros(size(TrainOPred));
%             T = cell(size(TrainOPred));
%             for t = 1:length(tSubs)
%                 for v = 1:8
%                     T{t,v} = predict(model,TrainFPred{t,v});
%                     acc(t,v) = mean((T{t,v}-TrainOPred{t,v}).^2);
%                 end
%             end
%             TrainAcc(s,nVisit) = mean(mean(acc,2));
%             Tr = predict(model,TrainFeatures);
%             TrainR2(s,nVisit) = corr2(TrainOutput,Tr)^2;
%             
%             % Test accuracy
%             acc = zeros(size(TestOPred));
%             C = cell(size(TestOPred));
%             for t = 1:length(nSubs)
%                 for v = 1:8
%                     C{t,v} = predict(model,TestFPred{t,v});
%                     acc(t,v) = mean((C{t,v}-TestOPred{t,v}).^2);
%                 end
%             end
%             TestAcc(s,nVisit) = mean(mean(acc,2));
%             Ts = predict(model,TestFeatures);
%             TestR2(s,nVisit) = corr2(TestOutput,Ts)^2;
%             
%             fprintf('Testing on K-Fold %d\t Visit %d .....\n\tTrain:\tMSE = %3.2f \tR2 = %1.6f\n\tTest:\tMSE = %3.2f \tR2 = %1.6f\n',s,nVisit,TrainAcc(s,nVisit),TrainR2(s,nVisit),TestAcc(s,nVisit),TestR2(s,nVisit));
            model = fitcecoc(TrainFeatures,TrainOutput,...
                'Coding','onevsall');
%             model = fitcsvm(TrainFeatures,TrainOutput,'Standardize',true,...
%                 'KernelFunction','rbf','KernelScale','auto','ScoreTransform','logit');
            
            % Training Accuracy
            acc = zeros(size(TrainOPred));
            for t = 1:length(tSubs)
                for v = 1:8
                    [T,Tscore] = predict(model,TrainFPred{t,v});
                    if wa
                        % have not tested yet ****
                        m = mean(Tscore(:,2))>=0.5;
                        acc(t,v) = mean(m==TrainOPred{t,v})>=0.5;
                    else
                        acc(t,v) = mean(T==TrainOPred{t,v})>=0.5;
                    end
                end
            end
            TrainAcc(s,nVisit) = mean(mean(acc,2));
            
            % Test Accuracy
            acc = zeros(size(TestOPred));
            for t = 1:length(nSubs)
                for v = 1:8
                    [C,Cscore] = predict(model,TestFPred{t,v});
                    if wa
                        m = mean(Cscore(:,2))>=0.5;
                        acc(t,v) = mean(m==TestOPred{t,v})>=0.5;
                    else
                        acc(t,v) = mean(C==TestOPred{t,v})>=0.5;
                    end
                end
            end
            TestAcc(s,nVisit) = mean(mean(acc,2));
            
            fprintf('Testing on K-Fold %d\t Visit %d .....\n\tTrain:\tAcc = %3.2f\n\tTest:\tAcc = %3.2f\n',s,nVisit,100*TrainAcc(s,nVisit),100*TestAcc(s,nVisit));
        
        else    % binary
            model = fitcsvm(TrainFeatures,TrainOutput,'Standardize',true,...
                'KernelFunction','rbf','KernelScale','auto','ScoreTransform','logit');
            
            % Training Accuracy
            acc = zeros(size(TrainOPred));
            for t = 1:length(tSubs)
                for v = 1:8
                    [T,Tscore] = predict(model,TrainFPred{t,v});
                    if wa
                        % have not tested yet ****
                        m = mean(Tscore(:,2))>=0.5;
                        acc(t,v) = mean(m==TrainOPred{t,v})>=0.5;
                    else
                        acc(t,v) = mean(T==TrainOPred{t,v})>=0.5;
                    end
                end
            end
            TrainAcc(s,nVisit) = mean(mean(acc,2));
            
            % Test Accuracy
            acc = zeros(size(TestOPred));
            for t = 1:length(nSubs)
                for v = 1:8
                    [C,Cscore] = predict(model,TestFPred{t,v});
                    if wa
                        m = mean(Cscore(:,2))>=0.5;
                        acc(t,v) = mean(m==TestOPred{t,v})>=0.5;
                    else
                        acc(t,v) = mean(C==TestOPred{t,v})>=0.5;
                    end
                end
            end
            TestAcc(s,nVisit) = mean(mean(acc,2));
            
            fprintf('Testing on K-Fold %d\t Visit %d .....\n\tTrain:\tAcc = %3.2f\n\tTest:\tAcc = %3.2f\n',s,nVisit,100*TrainAcc(s,nVisit),100*TestAcc(s,nVisit));
        
        end
    end
end

%% Save output
if sf
    if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
%         fclose(fileID);
        elt = toc;
        fnamemat = [fname,'MAT/',name,'.mat'];
        save(fnamemat,'TrainAcc','TrainR2','TestAcc','TestR2','elt','-v7.3');
    else
%         fclose(fileID);
        elt = toc;
        fnamemat = [fname,'MAT/',name,'.mat'];
        save(fnamemat,'TrainAcc','TestAcc','elt','-v7.3');
    end
else
%     fclose(fileID);
    elt = toc;
end


%% JUST FOR SAVING INTO TEXT FILE - must be outside of parallel loop
if sf
    for nVisit = vislist
        for s = 1:CVO.NumTestSets
            if strcmp(outclass,'valence')||strcmp(outclass,'arousal')
                fprintf(fileID,'Testing on K-Fold %d\t Visit %d .....\n\tTrain:\tMSE = %3.2f \tR2 = %1.6f\n\tTest:\tMSE = %3.2f \tR2 = %1.6f\n',s,nVisit,TrainAcc(s,nVisit),TrainR2(s,nVisit),TestAcc(s,nVisit),TestR2(s,nVisit));
            else
                fprintf(fileID,'Testing on K-Fold %d\t Visit %d .....\n\tTrain:\tAcc = %3.2f\n\tTest:\tAcc = %3.2f\n',s,nVisit,100*TrainAcc(s,nVisit),100*TestAcc(s,nVisit));
            end
        end
    end
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
    meanTrainAcc = mean(TrainAcc);
    stdTrainAcc = std(TrainAcc);
    meanTestAcc = mean(TestAcc);
    stdTestAcc = std(TestAcc);
    if sf
        fprintf(fileID,'\n\n-----\nVisit 1:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(1),100*stdTrainAcc(1));
        fprintf(fileID,'\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(1),100*stdTestAcc(1));
        fprintf(fileID,'\n\n-----\nVisit 2:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(2),100*stdTrainAcc(2));
        fprintf(fileID,'\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(2),100*stdTestAcc(2));
    end
    fprintf('\n\n-----\nVisit 1:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(1),100*stdTrainAcc(1));
    fprintf('\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(1),100*stdTestAcc(1));
    fprintf('\n\n-----\nVisit 2:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(2),100*stdTrainAcc(2));
    fprintf('\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(2),100*stdTestAcc(2));
else    % binary accuracy
    meanTrainAcc = mean(TrainAcc);
    stdTrainAcc = std(TrainAcc);
    meanTestAcc = mean(TestAcc);
    stdTestAcc = std(TestAcc);
    if sf
        fprintf(fileID,'\n\n-----\nVisit 1:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(1),100*stdTrainAcc(1));
        fprintf(fileID,'\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(1),100*stdTestAcc(1));
        fprintf(fileID,'\n\n-----\nVisit 2:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(2),100*stdTrainAcc(2));
        fprintf(fileID,'\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(2),100*stdTestAcc(2));
    end
    fprintf('\n\n-----\nVisit 1:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(1),100*stdTrainAcc(1));
    fprintf('\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(1),100*stdTestAcc(1));
    fprintf('\n\n-----\nVisit 2:\n\tMean Training Acc = %3.2f \t Stdev = %3.2f \n',100*meanTrainAcc(2),100*stdTrainAcc(2));
    fprintf('\tMean Testing  Acc = %3.2f \t Stdev = %3.2f \n',100*meanTestAcc(2),100*stdTestAcc(2));
end

if sf
    fclose(fileID);
end

end