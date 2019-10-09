function [FearFeatures, HappyFeatures] = featureExtraction(Fear,Happy,s,ov,feat)
% Function to extract all features used in analysis
% feat input: an array containing the features to include
%   1 - PSD
%   2 - FAA (must also include PSD)
%   3 - ChanCorr
%   4 - Bispec
%   5 - Bicoher
%   6 - Coh
%   7 - CPSD
%   8 - RMS
%
% Should include the following:
% power spectral density ///
% frontal alpha asymmetry ///
% bi-spectrum ///
% correlation coeff ///
% bi-coherence ///
% cross-spectrum ? cpsd
% cross-bispectrum ?
% coherence ///
% cross-bicoherence ?
% linear phase coupling ? ***
% quadratic phase coupling ? ***
% FBCSP ?

%% Combining Features

FearFeatures  = cell(size(Fear));
HappyFeatures = cell(size(Happy));

if sum(feat==1)>0
    % PSD features
    [FearPSD,HappyPSD] = getPSD(Fear,Happy,s,ov);
end
if sum(feat==2)>0
    % FAA features
    [FearFAA,HappyFAA] = getFAA(Fear,Happy,s,ov);
end
if sum(feat==3)>0
    % ChanCorr features
    [FearChancorr,HappyChancorr] = getChancorr(Fear,Happy,s,ov);
end
if sum(feat==4)>0
    % Bispec features
    [FearBispec,HappyBispec] = getBispec(Fear,Happy,s,ov);
end
if sum(feat==5)>0
    % Bicoher features
    [FearBicoher,HappyBicoher] = getBicoher(Fear,Happy,s,ov);
end
if sum(feat==6)>0
    % Coh features
    [FearCoh,HappyCoh] = getCoh(Fear,Happy,s,ov);
end
if sum(feat==7)>0
    % CPSD features
    [FearCPSD,HappyCPSD] = getCPSD(Fear,Happy,s,ov);
end
if sum(feat==8)>0
    % RMS features
    [FearRMS,HappyRMS] = getRMS(Fear,Happy,s,ov);
end

for nSubj = 1:size(Fear,1)
    for nVisit = 1:size(Fear,2)
        for nVid = 1:size(Fear,3)
            FearFeat = [];
            HappyFeat = [];
            if sum(feat==1)>0
                FearFeat  = cat(2,FearFeat,FearPSD{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyPSD{nSubj,nVisit,nVid});
            end
            if sum(feat==2)>0
                FearFeat  = cat(2,FearFeat,FearFAA{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyFAA{nSubj,nVisit,nVid});
            end
            if sum(feat==3)>0
                FearFeat = cat(2,FearFeat,FearChancorr{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyChancorr{nSubj,nVisit,nVid});
            end
            if sum(feat==4)>0
                FearFeat = cat(2,FearFeat,FearBispec{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyBispec{nSubj,nVisit,nVid});
            end
            if sum(feat==5)>0
                FearFeat = cat(2,FearFeat,FearBicoher{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyBicoher{nSubj,nVisit,nVid});
            end
            if sum(feat==6)>0
                FearFeat = cat(2,FearFeat,FearCoh{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyCoh{nSubj,nVisit,nVid});
            end
            if sum(feat==7)>0
                FearFeat = cat(2,FearFeat,FearCPSD{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyCPSD{nSubj,nVisit,nVid});
            end
            if sum(feat==8)>0
                FearFeat  = cat(2,FearFeat,FearRMS{nSubj,nVisit,nVid});
                HappyFeat = cat(2,HappyFeat,HappyRMS{nSubj,nVisit,nVid});
            end
            
            FearFeatures{nSubj,nVisit,nVid} = FearFeat;
            HappyFeatures{nSubj,nVisit,nVid} = HappyFeat;
        end
    end
end

end

function [FearPSD,HappyPSD] = getPSD(Fear,Happy,s,ov)
%% Calculating Power Spectral Density

try
    if ov == 0
        filename = sprintf('FH_S%d_PSD.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_PSD.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG PSD features\n\n');
    FearPSD  = calcPSD(Fear);
    HappyPSD = calcPSD(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_PSD.mat',s);
        save(savefile,'FearPSD','HappyPSD');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_PSD.mat',s,n(3:end));
        save(savefile,'FearPSD','HappyPSD');
    else
        error('Invalid input for OV');
    end
end

end

function [FearFAA,HappyFAA] = getFAA(Fear,Happy,s,ov)
%% Calculating Frontal Alpha Asymmetry
% % Depends on calcPSD

try
    if ov == 0
        filename = sprintf('FH_S%d_FAA.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_FAA.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
%     fprintf('Calculating new EEG FAA features\n\n');
%     FearFAA  = frontalAlphaAsym(FearPSD);
%     HappyFAA = frontalAlphaAsym(HappyPSD);
%     if ov == 0
%         savefile = sprintf('FH_S%d_FAA.mat',s);
%         save(savefile,'FearFAA','HappyFAA');
%     elseif (ov>0)&&(ov<1)
%         n = num2str(ov,2);
%         savefile = sprintf('FH_S%d_OVp%s_FAA.mat',s,n(3:end));
%         save(savefile,'FearFAA','HappyFAA');
%     else
        error('Invalid input for OV');
%     end
end

end

function [FearChancorr,HappyChancorr] = getChancorr(Fear,Happy,s,ov)
%% Calculating Correlation Coefficients

try
    if ov == 0
        filename = sprintf('FH_S%d_Chancorr.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_Chancorr.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG Chancorr features\n\n');
    FearChancorr  = chanCorr(Fear);
    HappyChancorr = chanCorr(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_Chancorr.mat',s);
        save(savefile,'FearChancorr','HappyChancorr');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_Chancorr.mat',s,n(3:end));
        save(savefile,'FearChancorr','HappyChancorr');
    else
        error('Invalid input for OV');
    end
end

end

function [FearCoh,HappyCoh] = getCoh(Fear,Happy,s,ov)
%% Calculating Coherence

try
    if ov == 0
        filename = sprintf('FH_S%d_Coh.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_Coh.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG Coh features\n\n');
    FearCoh  = chanCoh(Fear);
    HappyCoh = chanCoh(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_Coh.mat',s);
        save(savefile,'FearCoh','HappyCoh','-v7.3');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_Coh.mat',s,n(3:end));
        save(savefile,'FearCoh','HappyCoh','-v7.3');
    else
        error('Invalid input for OV');
    end
end
end

function [FearCPSD,HappyCPSD] = getCPSD(Fear,Happy,s,ov)
%% Calculating Cross-Power Spectral Density

try
    if ov == 0
        filename = sprintf('FH_S%d_CPSD.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_CPSD.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG CPSD features\n\n');
    FearCPSD  = chanCPSD(Fear);
    HappyCPSD = chanCPSD(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_CPSD.mat',s);
        save(savefile,'FearCPSD','HappyCPSD','-v7.3');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_CPSD.mat',s,n(3:end));
        save(savefile,'FearCPSD','HappyCPSD','-v7.3');
    else
        error('Invalid input for OV');
    end
end
end

function [FearBicoher,HappyBicoher] = getBicoher(Fear,Happy,s,ov)
%% Calculating Bicoherence Features

try
    if ov == 0
        filename = sprintf('FH_S%d_Bicoher.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_Bicoher.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG Bicoher features\n\n');
    FearBicoher  = bicoherEst(Fear);
    HappyBicoher = bicoherEst(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_Bicoher.mat',s);
        save(savefile,'FearBicoher','HappyBicoher','-v7.3');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_Bicoher.mat',s,n(3:end));
        save(savefile,'FearBicoher','HappyBicoher','-v7.3');
    else
        error('Invalid input for OV');
    end
end
end

function [FearBispec,HappyBispec] = getBispec(Fear,Happy,s,ov)
%% Calculating Bispectral Features
% % Takes 15 hours to calculate!!

try
    if ov == 0
        filename = sprintf('FH_S%d_Bispec.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_Bispec.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG Bispec features\n\n');
    FearBispec  = bispecEst(Fear);
    HappyBispec = bispecEst(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_Bispec.mat',s);
        save(savefile,'FearBispec','HappyBispec','-v7.3');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_Bispec.mat',s,n(3:end));
        save(savefile,'FearBispec','HappyBispec','-v7.3');
    else
        error('Invalid input for OV');
    end
end
end

function [FearRMS,HappyRMS] = getRMS(Fear,Happy,s,ov)
%% Calculating Power Spectral Density

try
    if ov == 0
        filename = sprintf('FH_S%d_RMS.mat',s);
        load(filename);
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        filename = sprintf('FH_S%d_OVp%s_RMS.mat',s,n(3:end));
        load(filename);
    else
        error('Invalid input for OV');
    end
catch
    fprintf('Calculating new EEG RMS features\n\n');
    FearRMS  = calcRMS(Fear);
    HappyRMS = calcRMS(Happy);
    if ov == 0
        savefile = sprintf('FH_S%d_RMS.mat',s);
        save(savefile,'FearRMS','HappyRMS');
    elseif (ov>0)&&(ov<1)
        n = num2str(ov,2);
        savefile = sprintf('FH_S%d_OVp%s_RMS.mat',s,n(3:end));
        save(savefile,'FearRMS','HappyRMS');
    else
        error('Invalid input for OV');
    end
end

end