function [features] = bicoherEst(input)
% This function will compute the bicoherence using the HOSA toolbox
% function bicoher.
%  
% bicoher - Direct (FD) method for estimating bicoherence
%  	[bic,waxis] = bicoher (y,  nfft, wind, segsamp, overlap)
%  	y     - data vector or time-series
%  	nfft - fft length [default = power of two > segsamp]
%  	       actual size used is power of two greater than 'nsamp'
%  	wind - specifies the time-domain window to be applied to each
%  	       data segment; should be of length 'segsamp' (see below);
%  		otherwise, the default Hanning window is used.
%  	segsamp - samples per segment [default: such that we have 8 segments]
%  	        - if x is a matrix, segsamp is set to the number of rows
%  	overlap - percentage overlap, allowed range [0,99]. [default = 50];
%  	        - if x is a matrix, overlap is set to 0.
%  	bic     - estimated bicoherence: an nfft x nfft array, with origin
%  	          at the center, and axes pointing down and to the right.
%  	waxis   - vector of frequencies associated with the rows and columns
%  	          of bic;  sampling frequency is assumed to be 1.

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
                    [bic,waxis] = bicoher(y);
                    nmin = floor(length(waxis)/2)+1;
                    nmax = length(waxis)-nmin+1+ceil(nmin/2);
                    pup = triu(bic(nmin:nmax,nmin:nmax));
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
                    bicoher_sq_sum = [sum(p_delta_delta.^2),...
                        sum(p_theta_delta.^2),...
                        sum(p_alpha_delta.^2),...
                        sum(p_beta_delta.^2),...
                        sum(p_theta_theta.^2),...
                        sum(p_alpha_theta.^2),...
                        sum(p_beta_theta.^2),...
                        sum(p_alpha_alpha.^2),...
                        sum(p_beta_alpha.^2),...
                        sum(p_beta_beta.^2)];
                    chanfeat(chan,:) = bicoher_sq_sum;
                end
                features{nSubj,nVisit,nVid}(nSeg,:) = chanfeat(:)';
            end
        end
    end
end


end