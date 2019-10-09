function [features] = calcPSD(input)
% This function will use the EEGLab spectopo function to calculate the
% Power Spectral Density of the EEG signal using pwelch.
%
% data input required: 
% input should be Fear or Happy 
% an nSubj x nVisit x nVid cell containing
% nSeg x 1 cells arrays
% (eg. max is Fear{116,2,4}{7,1} for FearHappy_S30_OVp5.mat)
%
% The default settings used will be:
% frames = 0 (i.e. data length)
% srate = 256
% 'freqfac' = 4 (i.e. ntimes to oversample (to adjust frequency resolution)
%   note: this things doesn't seem to be doing anything...
% 'plot' = 'off'

features = cell(size(input));
Fs = 256;
% ch = 'LallemandeCap.locs';
nchans = 5;

for nSubj = 1:size(input,1)
    for nVisit = 1:size(input,2)
        for nVid = 1:size(input,3)
            features{nSubj,nVisit,nVid} = zeros(size(input{nSubj,nVisit,nVid},1),25);
            for nSeg = 1:size(input{nSubj,nVisit,nVid},1)
                data = input{nSubj,nVisit,nVid}{nSeg,1}';
                
                [spectra,freqs]= spectopo(data,...
                    0, Fs, 'freqfac', 4, 'plot', 'off');
                
                freqs = freqs';     % keep for checks

                % Store in array
                % spectra (nchans, nfreqs) contains the power spectra in dB
                % freqs = frequencies of spectra
                % freqs is the same for every video where
                delta_idx  = 2:16;    %(<4 Hz)
                theta_idx  = 17:32;   %(4-7 Hz)
                alpha_idx  = 33:64;   %(8-15 Hz)
                beta_idx   = 65:128;  %(16-31 Hz)
                gamma_idx  = 128:401; %(32-100 Hz)
                
                % Converting back form dB
                for j = 1:length(spectra)
                    for i = 1:5
                        spectra(i,j)=spectra(i,j)/10;
                        spectra(i,j) = 10^spectra(i,j);
                        
                    end
                end
                
                %Calculate Power in Bands
                deltaPower = zeros(1,nchans);
                thetaPower = zeros(1,nchans);
                alphaPower = zeros(1,nchans);
                betaPower  = zeros(1,nchans);
                gammaPower = zeros(1,nchans);
                for electrode = 1:nchans
                    deltaPower(electrode) = mean(spectra(electrode,delta_idx));
                    thetaPower(electrode) = mean(spectra(electrode,theta_idx));
                    alphaPower(electrode) = mean(spectra(electrode,alpha_idx));
                    betaPower(electrode)  = mean(spectra(electrode,beta_idx));
                    gammaPower(electrode) = mean(spectra(electrode,gamma_idx));
                end
                
                features{nSubj,nVisit,nVid}(nSeg,:) = ...
                    [deltaPower, thetaPower, alphaPower, betaPower, gammaPower];
            end
        end
    end
end

end