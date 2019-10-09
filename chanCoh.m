function [features] = chanCoh(input)
% Calculates coherence between channels for each segment wrt bands

features = cell(size(input));

for nVisit = 1:size(input,2)
    for nVid = 1:size(input,3)
        parfor nSubj = 1:size(input,1)
            fprintf('\nCoh - subj %d \tvis %d\t vid %d',nSubj,nVisit,nVid);
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),50);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                data = input{nSubj,nVisit,nVid}{nSeg,1};
                C = zeros(129,10);
                C(:,1) = mscohere(data(:,1),data(:,2),hamming(128),80,256,256);
                C(:,2) = mscohere(data(:,1),data(:,3),hamming(128),80,256,256);
                C(:,3) = mscohere(data(:,1),data(:,4),hamming(128),80,256,256);
                C(:,4) = mscohere(data(:,1),data(:,5),hamming(128),80,256,256);
                C(:,5) = mscohere(data(:,2),data(:,3),hamming(128),80,256,256);
                C(:,6) = mscohere(data(:,2),data(:,4),hamming(128),80,256,256);
                C(:,7) = mscohere(data(:,2),data(:,5),hamming(128),80,256,256);
                C(:,8) = mscohere(data(:,3),data(:,4),hamming(128),80,256,256);
                C(:,9) = mscohere(data(:,3),data(:,5),hamming(128),80,256,256);
                C(:,10) = mscohere(data(:,4),data(:,5),hamming(128),80,256,256);
                coh = zeros(5,10);
                for i = 1:10
                    coh(1,i) = mean(C(1:4,i))/length(1:4);
                    coh(2,i) = mean(C(5:8,i))/length(5:8);
                    coh(3,i) = mean(C(9:16,i))/length(9:16);
                    coh(4,i) = mean(C(17:32,i))/length(17:32);
                    coh(5,i) = mean(C(33:129,i))/length(33:129);
                end
                features{nSubj,nVisit,nVid}(nSeg,:) = coh(:)';
            end
        end
    end
end

end