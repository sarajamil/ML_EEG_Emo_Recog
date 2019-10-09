function [FeatSel,fs,CVO] = loadFSKfold(feat,miq,s,ov,k)
% function will load FS list if it is saved
% otherwise it will return a FeatSel of zeros and fs = true
% fs is used to know whether it is necessary to do mRMR 
% 
% naming convention for FS lists:
% 'FS_'(MID or MIQ)'_'(PSD and/or FAA and/or CC and/or BIS and/or BIC
% and/or COH and/or CPSD)'.mat'
% in that order

try
    % try to load list
    if ov>0
        n = num2str(ov,2);
        fstr = sprintf('FS_%dFold_S%d_OVp%s_',k,s,n(3:end));
    else
        fstr = sprintf('FS_%dFold_S%d_',k,s);
    end
    
    if miq
        fstr = strcat(fstr,'MIQ_');
    else
        fstr = strcat(fstr,'MID_');
    end
    
    if sum(feat==1)>0, fstr = strcat(fstr,'PSD'); end
    if sum(feat==2)>0, fstr = strcat(fstr,'FAA'); end
    if sum(feat==3)>0, fstr = strcat(fstr,'CC'); end
    if sum(feat==4)>0, fstr = strcat(fstr,'BIS'); end
    if sum(feat==5)>0, fstr = strcat(fstr,'BIC'); end
    if sum(feat==6)>0, fstr = strcat(fstr,'COH'); end
    if sum(feat==7)>0, fstr = strcat(fstr,'CPSD'); end
    if sum(feat==8)>0, fstr = strcat(fstr,'RMS'); end
    
    fstr = strcat(fstr,'.mat');
    load(fstr); %check directory
    fs = false;
catch
    % no file to load
    FeatSel = 0;
    fs = true;
    CVO = 0;
end

end