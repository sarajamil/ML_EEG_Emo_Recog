function [subjClean] = removeSubjects(Fear, Happy, p)
% This function is meant to remove the worst subjects data from the entire
% group for LOO classification/regression or exclusion from Within-subj
% analyses. It works by calculating the RMS values of the data and then
% removing the top p portion of the subjects with the highest RMS values.
% Note: p is given as a number between [0,1] and is usually 0.1 or 0.15

% Load the data and split into time windows
s = 30;     % window size in seconds
ov = 0.5;     % overlap b/t (0,1)

if (size(who('Fear'),1)==0)||(size(who('Happy'),1)==0)
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
        fprintf('Creating new segmented EEG data...\n\n');
        if ov==0
            [Fear, Happy] = dataSplitNonOV(s);
        elseif (ov>0)&&(ov<1)
            [Fear, Happy] = dataSplitOV(s,ov);
        end
    end
end


FearRMS = zeros(size(Fear));
HappyRMS = zeros(size(Happy));

for nSubj = 1:size(Fear,1)
    for nVisit = 1:size(Fear,2)
        for nVid = 1:size(Fear,3)
            Fdata = Fear{nSubj,nVisit,nVid};
            fRMS = zeros(length(Fdata),1);
            for nSeg = 1:length(Fdata)
                fRMS(nSeg,1) = mean(rms(Fdata{nSeg,1}));
            end
            FearRMS(nSubj,nVisit,nVid) = mean(fRMS);
            Hdata = Happy{nSubj,nVisit,nVid};
            hRMS = zeros(length(Hdata),1);
            for nSeg = 1:length(Hdata)
                hRMS(nSeg,1) = mean(rms(Hdata{nSeg,1}));
            end
            HappyRMS(nSubj,nVisit,nVid) = mean(hRMS);
        end
    end
end

vidmean(:,:,1) = mean(FearRMS,3); 
vidmean(:,:,2) = mean(HappyRMS,3);
allvidmean = mean(vidmean,3);
subjmean = mean(allvidmean,2);
[~,ind] = sort(subjmean);

n = size(Fear,1) - ceil(size(Fear,1)*p);
subjClean = ind(1:n);

end



