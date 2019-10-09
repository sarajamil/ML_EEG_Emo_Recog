function [features] = bispecEst(input)
% This function will compute the bispectrum estimate using direct methods. 
% This function will use the HOSA toolbox function bispecd.
% 
%  	[Bspec,waxis] = bispecd (y,  nfft, wind, segsamp, overlap)
%  	y    - data vector or time-series
%  	nfft - fft length [default = power of two > segsamp]
%  	wind - window specification for frequency-domain smoothing
%  	       if 'wind' is a scalar, it specifies the length of the side
%  	          of the square for the Rao-Gabr optimal window  [default=5]
%  	       if 'wind' is a vector, a 2D window will be calculated via
%  	          w2(i,j) = wind(i) * wind(j) * wind(i+j)
%  	       if 'wind' is a matrix, it specifies the 2-D filter directly
%  	segsamp - samples per segment [default: such that we have 8 segments]
%  	        - if y is a matrix, segsamp is set to the number of rows
%  	overlap - percentage overlap [default = 50]
%  	        - if y is a matrix, overlap is set to 0.
%
% If y is a matrix, the columns are assumed to correspond to independent
% realizations; in this case overlap is set to zero, and samp_seg is set to 
% the row dimension
% 
% bspec is the estimated bispectrum; it is an nfft-by-nfft array, with 
% origin at the center, and axes pointing down and to the right.
% 
% waxis is the set of frequencies associated with the bispectrum in bspec. 
% Thus, the ith row (or column) of bspec corresponds to the frequency 
% waxis(i), i=1,...,nfft. Frequencies are normalized; that is, the sampling 
% frequency is assumed to be unity.

features = cell(size(input));
Fs = 256;

for nSubj = 1:size(input,1)
    for nVisit = 1:size(input,2)
        for nVid = 1:size(input,3)
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),50);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                data = input{nSubj,nVisit,nVid}{nSeg,1};
                chanfeat = zeros(5,10);
                parfor chan = 1:5
                    y = data(:,chan);
                    % [bspec,waxis] = bispecd(y,length(y),5,length(y),0);
                    % this takes WAAAY too long (ie. 2.5 DAYS vs 5.5 hours)
                    % default:
                    % nfft = 2048
                    % nsamp = 1706
                    % overlap = 853
                    [bspec,waxis] = bispecd(y);
                    nmin = floor(length(waxis)/2)+1;
                    nmax = length(waxis)-nmin+1+ceil(nmin/2);
                    pup = triu(bspec(nmin:nmax,nmin:nmax));
                    waxis = waxis(nmin:nmax).*Fs;
                    id1 = sum(waxis<4);
                    id2 = sum(waxis<8);
                    id3 = sum(waxis<16);
                    id4 = sum(waxis<32);
                    delta = 1:id1;
                    theta = (id1+1):id2;
                    alpha = (id2+1):id3;
                    beta  = (id3+1):id4;
                    p_delta_delta = pup(delta,delta);
                    p_theta_delta = pup(delta,theta);
                    p_alpha_delta = pup(delta,alpha);
                    p_beta_delta  = pup(delta,beta);
                    p_theta_theta = pup(theta,theta);
                    p_alpha_theta = pup(theta,alpha);
                    p_beta_theta  = pup(theta,beta);
                    p_alpha_alpha = pup(alpha,alpha);
                    p_beta_alpha  = pup(alpha,beta);
                    p_beta_beta   = pup(beta,beta);
                    p_delta_delta = abs(p_delta_delta(triu(true(size(p_delta_delta)))));
                    p_theta_delta = abs(p_theta_delta(:));
                    p_alpha_delta = abs(p_alpha_delta(:));
                    p_beta_delta  = abs(p_beta_delta(:));
                    p_theta_theta = abs(p_theta_theta(triu(true(size(p_theta_theta)))));
                    p_alpha_theta = abs(p_alpha_theta(:));
                    p_beta_theta  = abs(p_beta_theta(:));
                    p_alpha_alpha = abs(p_alpha_alpha(triu(true(size(p_alpha_alpha)))));
                    p_beta_alpha  = abs(p_beta_alpha(:));
                    p_beta_beta   = abs(p_beta_beta(triu(true(size(p_beta_beta)))));
%                     bispec_sum = [sum(p_delta_delta),...
%                         sum(p_theta_delta),...
%                         sum(p_alpha_delta),...
%                         sum(p_beta_delta),...
%                         sum(p_theta_theta),...
%                         sum(p_alpha_theta),...
%                         sum(p_beta_theta),...
%                         sum(p_alpha_alpha),...
%                         sum(p_beta_alpha),...
%                         sum(p_beta_beta),];
                    bispec_sq_sum = [sum(p_delta_delta.^2),...
                        sum(p_theta_delta.^2),...
                        sum(p_alpha_delta.^2),...
                        sum(p_beta_delta.^2),...
                        sum(p_theta_theta.^2),...
                        sum(p_alpha_theta.^2),...
                        sum(p_beta_theta.^2),...
                        sum(p_alpha_alpha.^2),...
                        sum(p_beta_alpha.^2),...
                        sum(p_beta_beta.^2),];
                    chanfeat(chan,:) = bispec_sq_sum;
                end
                features{nSubj,nVisit,nVid}(nSeg,:) = chanfeat(:)';
            end
        end
    end
end


end