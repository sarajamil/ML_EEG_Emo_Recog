function [FearOut, HappyOut] = outputClassAll(Fear, Happy, type)
% Defines the output class of the data
% for input into SVM
% 
% Input:
% Fear - 3x3 cell (nSubj,nVisit,nVid) of nSeg's
% Happy - 3x3 cell (nSubj,nVisit,nVid) of nSeg's
% type - 
%   'binary' outputs fear=1, happy=0
%   'valence' outputs subjects valence rating [1,7]
%   'arousal' outputs subjects arousal rating [1,7]
%   no input results in binary output classes
% 
% Dependencies:
% preprocessOutputs.m (for valence and arousal only)

if nargin < 3
    type = 'unspecified';
end

switch type
    case 'binary'
        [FearOut, HappyOut] = binaryOutputClass(Fear, Happy);
    case 'valence'
        [FearOut, HappyOut] = valenceOutputClass(Fear, Happy);
    case 'arousal'
        [FearOut, HappyOut] = arousalOutputClass(Fear, Happy);
    otherwise
        fprintf('No output type specified. Using binary output...\n\n');
        [FearOut, HappyOut] = binaryOutputClass(Fear, Happy);
end

end

function [FearOut, HappyOut] = binaryOutputClass(Fear, Happy)
FearOut = cell(size(Fear));
HappyOut = cell(size(Happy));

for nSubj = 1:size(Fear,1)
    for nVisit = 1:size(Fear,2)
        for nVid = 1:size(Fear,3)
            FearOut{nSubj,nVisit,nVid} = ones(size(Fear{nSubj,nVisit,nVid},1),1);
        end
    end
end

for nSubj = 1:size(Happy,1)
    for nVisit = 1:size(Happy,2)
        for nVid = 1:size(Happy,3)
            HappyOut{nSubj,nVisit,nVid} = zeros(size(Happy{nSubj,nVisit,nVid},1),1);
        end
    end
end

end

function [FearOut, HappyOut] = valenceOutputClass(Fear, Happy)
FearOut = cell(size(Fear));
HappyOut = cell(size(Happy));

[num,~] = preprocessOutputs(); % 2:17
fnum = num(:,2:9);
hnum = num(:,10:17);

for nSubj = 1:size(Fear,1)
    for nVisit = 1:size(Fear,2)
        for nVid = 1:size(Fear,3)
            m = size(Fear{nSubj,nVisit,nVid},1);
            FearOut{nSubj,nVisit,nVid} = fnum(nSubj,2*nVid+nVisit-2)*ones(m,1);
        end
    end
end

for nSubj = 1:size(Happy,1)
    for nVisit = 1:size(Happy,2)
        for nVid = 1:size(Happy,3)
            m = size(Happy{nSubj,nVisit,nVid},1);
            HappyOut{nSubj,nVisit,nVid} = hnum(nSubj,2*nVid+nVisit-2)*ones(m,1);
        end
    end
end

end

function [FearOut, HappyOut] = arousalOutputClass(Fear, Happy)
FearOut = cell(size(Fear));
HappyOut = cell(size(Happy));

[num,~] = preprocessOutputs(); % 18:33
fnum = num(:,18:25);
hnum = num(:,26:33);

for nSubj = 1:size(Fear,1)
    for nVisit = 1:size(Fear,2)
        for nVid = 1:size(Fear,3)
            m = size(Fear{nSubj,nVisit,nVid},1);
            FearOut{nSubj,nVisit,nVid} = fnum(nSubj,2*nVid+nVisit-2)*ones(m,1);
        end
    end
end

for nSubj = 1:size(Happy,1)
    for nVisit = 1:size(Happy,2)
        for nVid = 1:size(Happy,3)
            m = size(Happy{nSubj,nVisit,nVid},1);
            HappyOut{nSubj,nVisit,nVid} = hnum(nSubj,2*nVid+nVisit-2)*ones(m,1);
        end
    end
end

end
