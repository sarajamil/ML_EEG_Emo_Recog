function [features] = chanCorr(input)
% Calculates correlation between channels for each segment

features = cell(size(input));

for nSubj = 1:size(input,1)
    for nVisit = 1:size(input,2)
        for nVid = 1:size(input,3)
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),10);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                data = input{nSubj,nVisit,nVid}{nSeg,1};
                chancorr = corrcoef(data);
                chancorr = chancorr(triu(true(size(chancorr)),1))';
                features{nSubj,nVisit,nVid}(nSeg,:) = chancorr;
            end
        end
    end
end


end