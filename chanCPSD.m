function [features] = chanCPSD(input)
% Calculates coherence between channels for each segment wrt bands

features = cell(size(input));

for nVisit = 1:size(input,2)
    for nVid = 1:size(input,3)
        parfor nSubj = 1:size(input,1)
            fprintf('\nCPSD - subj %d \tvis %d\t vid %d',nSubj,nVisit,nVid);
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),50);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                data = input{nSubj,nVisit,nVid}{nSeg,1};
                perms = nchoosek(1:5,2);
                C = zeros(5,10);
                for i = 1:10
                    P1 = cpsd(data(:,perms(i,1)),data(:,perms(i,2)),hamming(128),80,0:3,256);
                    P2 = cpsd(data(:,perms(i,1)),data(:,perms(i,2)),hamming(128),80,4:7,256);
                    P3 = cpsd(data(:,perms(i,1)),data(:,perms(i,2)),hamming(128),80,8:15,256);
                    P4 = cpsd(data(:,perms(i,1)),data(:,perms(i,2)),hamming(128),80,16:31,256);
                    P5 = cpsd(data(:,perms(i,1)),data(:,perms(i,2)),hamming(128),80,32:100,256);
                    C(1,i) = mean(abs(P1));
                    C(2,i) = mean(abs(P2));
                    C(3,i) = mean(abs(P3));
                    C(4,i) = mean(abs(P4));
                    C(5,i) = mean(abs(P5));
                end
                features{nSubj,nVisit,nVid}(nSeg,:) = C(:)';
            end
        end
    end
end

end