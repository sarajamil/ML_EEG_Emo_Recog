function [features] = frontalAlphaAsym(input)
% Calculates frontal alpha asymmetry
% FAA = log(alpha power right F4/alpha power left F3)
% 
% This function will use the output of calcPSD as input
% (eg. FearPSD{116,2,4} is an 11x25 matrix - nSeg x nPowerBands)
% 
% channel locations:
% 1	       0	   0.275	      Fz
% 2	     -39	 0.35611	      F3
% 3	      39	 0.35611	      F4
% 4	     180	 0.38333	     POz
% 5	       0	   0	          Cz

features = cell(size(input));

for nSubj = 1:size(input,1)
    for nVisit = 1:size(input,2)
        for nVid = 1:size(input,3)
            fpsd = input{nSubj,nVisit,nVid};
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),5);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                % Calculate FAA
                F3delta = fpsd(nSeg,2);
                F4delta = fpsd(nSeg,3);
                F3theta = fpsd(nSeg,7);
                F4theta = fpsd(nSeg,8);
                F3alpha = fpsd(nSeg,12);
                F4alpha = fpsd(nSeg,13);
                F3beta  = fpsd(nSeg,17);
                F4beta  = fpsd(nSeg,18);
                F3gamma = fpsd(nSeg,22);
                F4gamma = fpsd(nSeg,23);
                
                FAA = [log(F4delta/F3delta), log(F4theta/F3theta), log(F4alpha/F3alpha),...
                    log(F4beta/F3beta), log(F4gamma/F3gamma)];
                features{nSubj,nVisit,nVid}(nSeg,:) = FAA;
            end
        end
    end
end

end
