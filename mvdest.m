% Jan. 01, 2022

% Aref Majdara

% Multivariate Density Estimation using BSP
% Gets a NxD input file, and estimates the corresponding probability
% density function by making sequentioanl cuts, based on BSP algorithm.
%
% Reference for basic BSP method:
% Luo Lu and Hui Jiang and Wing H. Wong, "Multivariate Density Estimation
% by Bayesian Sequential Partitioning", Journal of the American Statistical
% Association, 2013.

% Reference for online density estimation using BSP:
% A. Majdara and S. Nooshabadi, "Online Density Estimation Over
% High-Dimensional Stationary and Non-stationary Data Streams", Data &
% Knowledge Egineering, vol. 123, 2019.

% Input Parameters
%   problemType: densityEstimation / classification
%   progressive: Enable/Disable progressive update of the partitions
%   copulaEnbl:  Enable/Disable copula transform for multivariate problems
%   KLDivEnbl:   Calculate KL divergence?
%   N0:          Desired block size of data
%   MMarginal:   Number of smaple partitions for marginal calculations
%   MCopula:     Number of sample partitions for copula domain calculations
%   bestCutMarginal:
%                If set true, in marginals, at each level j, the potential
%                cut with the highest posterior probability is picked. If
%                false, the cut is ransdomly generated from the PMF.
%   bestCutCopula:
%                Same as above, for copula part.

% Outputs:
%   KLD:         KL divergence between estimated and actual densities
%   Time:        Execution time (marginals/copula)
%   Cuts:        Number of cuts made in each marginal or copula domain
%   NMSE:        Normalized Mean Squared Error
%   MISE:        Mean Integrated Squared Error
%   HLGR:        Hellinger distance


% mvdest calls the 'bincut' function for each block of data, to get the
% coordinates of the subregions (histogram bins) and their corresponding
% estimated local densities. These information are passed to  returned
% partitions and estimated densities to measure the divergencec/distance between
% the provided actual densities



function mvdest
close all
clear
clc

simulationID='D2_N20k'
outputTextFile=[simulationID,'.txt']
outputFileName=[simulationID,'.mat']

diary(outputTextFile)

warning('off', 'all');
yes=true;
no=false;

%% BSP Parameters

progressive=no
blockAveraging=no
streaming=no              

% Settings for online estimation
NchVector=[110000, 55000, 90000, 45000, 70000, 50000, 80000];  
NchVectorCum=cumsum(NchVector);      
cycles=length(NchVector);  
nTest=700000;              
nTestSub=nTest/(cycles);   
weights_8=[1 3 5 7 9 11 13 15];    % Averaging weights
B=1;              

generatePlots=yes      % Generate plots for 2D cases

copulaEnbl=no          % Copula or direct?
KLDivEnbl=yes          % Select 'yes' only if true density is available

distType='gaussian';   % 'gaussian', 'beta', 'simpson'


% Addresses to training data, test data, and true densities 
inputFile = 'MVN_N200000_D2'
testDataFileName = 'MVN_N150000_D2'
testDataDensitiesFileName = 'MVN_N150000_D2_densities'


numberOfRuns=1          % If no blocking, set to 1

% M for marginal and copula parts
MMarginal=200
MMarginalProg=200
MCopula=200
bestCutMarginal=no
bestCutMarginalProg=no
bestCutCopula=no

jMaxMarginal=4000 %300    % Max number of marginal cuts
jMaxCopula=4000   %300    % Max number of copula cuts

resmpMarginal=no;
resmpCopula=no;
resmpInterval=4;

problemType='densityEstimation';
% problemType='classification';

simulationType='serial';
%simulationType='parallel';

marCutsMethod='binary';
% marCutsMethod='median';

kdeForMarginals=no;

fileName='allParameters';
save(fileName);


%% Load input files

N0=20000             % Block size

load(inputFile);
allData = data(:,:);

N=size(data,1)
D=size(data,2)

load(testDataFileName);
testData = data;
allBounds(1,:)=min(min(allData),min(testData));
allBounds(2,:)=max(max(allData),max(testData));
load(testDataDensitiesFileName);

nTest=size(testData, 1);

%% Variable declarations

KLD=0; NMSE=0; MISE=0; HLGR=0;

if(streaming)
    allDensities=zeros(nTestSub, numberOfRuns);
    avgDensities=zeros(nTestSub, numberOfRuns);
    
    densMAPsBlk=zeros(jMaxMarginal, D, numberOfRuns);
    VkMAPsBlk=zeros(jMaxMarginal, D, numberOfRuns);
    bestCoordsMAPsBlk=zeros(2, jMaxMarginal, D, numberOfRuns);
    marginalCutsBlk=zeros(D, numberOfRuns);
    xtMAPcopulaBlk=zeros(2*D,jMaxCopula,numberOfRuns);
    densMAPcopulaBlk=zeros(jMaxCopula, numberOfRuns);
    copulaCutsBlk=zeros(numberOfRuns,1);
    
else
    allDensities=zeros(nTest, numberOfRuns);
    avgDensities=zeros(nTest, numberOfRuns);
end


KLDs = zeros(numberOfRuns+2, 1);       
NMSEs = zeros(numberOfRuns+2, 1);
MISEs = zeros(numberOfRuns+2, 1);
HLGRs = zeros(numberOfRuns+2, 1);

trainStageTimes = zeros(numberOfRuns+3, 1);
marginalCuts = zeros(D+4, numberOfRuns+3);
copulaCuts = zeros(numberOfRuns+3,1);
marginalsTime = zeros(numberOfRuns+3, 1);
copulaTime = zeros(numberOfRuns+3, 1);

testStageTimes = zeros(numberOfRuns+3, 1);



%% Setup Parpool (if needed)

if (strcmp(simulationType ,'parallel'))
    
    ver = version('-release');
    numberOfCores=feature('numcores');
    poolSize=numberOfCores;
    
    s=0;
    if (ver=='2012b')
        s = matlabpool('size');
        if (s~=poolSize)
            if (s~=0) matlabpool close; end
            matlabpool(poolSize)
        end
    else
        poolobj = gcp('nocreate'); % If no pool, do not create new one.
        p=0;
        if (~ isempty (poolobj))
            p=poolobj.NumWorkers;
        end
        if (p ~= poolSize)
            delete(gcp('nocreate'));
            parpool('local', poolSize)
        end
    end
end

%% Delete files from previous run

delete('allVariables.mat')
delete('allVars*.mat')
delete('best_*.mat')
delete('tree*.mat')

%% Divide the dataset into chunks

for i=1:numberOfRuns
    
    i
    trainData=allData((i-1)*N0+1:i*N0,:);
    t0=tic;
    
    if ((progressive==no) || (i==1))
        
        [bestTrees, densMAPs, VkMAPs, bestCoordsMAPs, marginalCdfs, margCuts, xtMAPcopula, densMAPcopula, copCuts, tMarginals, tCopula]=...
            bincut(i, allBounds, trainData, copulaEnbl, resmpMarginal, resmpCopula, resmpInterval, jMaxMarginal, MMarginal, jMaxCopula, MCopula, bestCutMarginal, bestCutCopula);
    else
        trainData=allData((i-1)*N0+1:i*N0,:);
        
        [bestTrees, densMAPs, VkMAPs, bestCoordsMAPs, margCuts, xtMAPcopula, densMAPcopula, copCuts, tMarginals, tCopula]=...
            bincut_progressive(i, allBounds, trainData, copulaEnbl, resmpMarginal, resmpCopula, resmpInterval, jMaxMarginal, MMarginalProg, jMaxCopula, MCopula, bestCutMarginalProg, bestCutCopula);
    end
    
    executionTime=toc(t0);
    

    if (copulaEnbl)
        for d=1:D
            load(['best_d', int2str(d)]);
            eval(['best_d', int2str(d), '=x;']);
        end
    
    blocksBestTreesName=['bestMarginalTrees_blk_' int2str(i)];
    save(blocksBestTreesName, 'best_d*');
    clear('best_d*')

    end

    densMAPsBlk(:,:,i)=densMAPs;
    VkMAPsBlk(:,:, i)=VkMAPs;
    bestCoordsMAPsBlk(:,:,:, i)=bestCoordsMAPs;
    marginalCutsBlk(:,i)=margCuts;
    xtMAPcopulaBlk(:,1:copCuts,i)=xtMAPcopula;
    densMAPcopulaBlk(:,i)=densMAPcopula;
    copulaCutsBlk(i)=copCuts;
    
    t0TestStage= tic;
    
    if ((KLDivEnbl==yes))
        
        t0TestStage=tic;


        actDens=actualDensities;
        
        testDataSub=testData;  

        if(streaming)
            n=i*N0;
            kk = n > NchVectorCum;
            k=sum(kk)+1;
            testDataSub=testData((k-1)*nTestSub+1:k*nTestSub,:);
            actDens=actDens((k-1)*nTestSub+1:k*nTestSub,:);
        end     
         
            W=min(i, B);
            weights=weights_8((B-W+1:end));
            
          

            for ii=(i-W+1):i
                if (copulaEnbl==yes)
                    blocksBestTreesName=['bestMarginalTrees_blk_' int2str(ii)];
                    estDens= estimate_copula(blocksBestTreesName, testDataSub, densMAPsBlk(:,:,ii), VkMAPsBlk(:,:, ii), bestCoordsMAPsBlk(:,:,:, ii),marginalCutsBlk(:,ii), xtMAPcopulaBlk(:,:,ii), densMAPcopulaBlk(:,ii), copulaCutsBlk(ii));
                else
                    estDens= estimate_direct(bestTrees, testDataSub, xtMAPcopula, densMAPcopula, copCuts);
                end
                allDensities(:,ii) = estDens;
            end

        
        
        if (blockAveraging)
            if (streaming)
                    s=sum(weights);
                    for r=1:size(avgDensities,1)
                        avgDensities(r,i) = (1/s) * sum( (weights .* allDensities(r,i-W+1:i)),2 );
                    end
            else
                avgDensities(:,i) = mean(allDensities(:,1:i),2);
            end

            estDens=avgDensities(:,i);
        end
        
        actDens = actDens(estDens~=0);
        estDens = estDens(estDens~=0);
        
        KLD= mean (log(actDens./estDens));
        NMSE = mean( ((estDens-actDens).^2) / ( mean(estDens) * mean(actDens) ) );
        MISE = mean( ((estDens-actDens).^2) / ( mean(estDens) * mean(actDens) ) );
        HLGR = 1- mean( sqrt(estDens./actDens) );
    end
    

    testStageTimes(i,1)=toc(t0TestStage);
    
    KLDs(i,1)=KLD;
    NMSEs(i,1)=NMSE;
    MISEs(i,1)=MISE;
    HLGRs(i,1)=HLGR;
    
    marginalCuts(1:end-4,i) = margCuts;
    marginalCuts(D+2,i) = sum(marginalCuts(1:D,i));
    marginalCuts(end-1,i) = mean(marginalCuts(1:D,i));
    marginalCuts(end,i) = std(marginalCuts(1:D,i));
    
    copulaCuts(i,1)=copCuts;
    
    trainStageTimes(i,1) = executionTime;
    marginalsTime(i,1) = tMarginals;
    copulaTime(i,1) = tCopula;
    
    save(outputFileName, 'allDensities', 'avgDensities', 'KLDs', 'NMSEs', 'MISEs', 'HLGRs', 'marginalCuts', 'copulaCuts', 'trainStageTimes', 'marginalsTime', 'copulaTime', 'testStageTimes');
    
    if(generatePlots)
        if (~ KLDivEnbl)
            testDataSub=trainData;
        end
        plots(trainData, testDataSub, actualDensities, marginalCdfs, margCuts, copCuts, densMAPs, VkMAPs, bestCoordsMAPs, xtMAPcopula, densMAPcopula)
    end
       
end

%%

KLDs(numberOfRuns+2,1) = mean(KLDs(1:numberOfRuns,1));
KLDs(numberOfRuns+3,1) =  std(KLDs(1:numberOfRuns,1));

NMSEs(numberOfRuns+2,1) = mean(NMSEs(1:numberOfRuns,1));
NMSEs(numberOfRuns+3,1) =  std(NMSEs(1:numberOfRuns,1));

MISEs(numberOfRuns+2,1) = mean(MISEs(1:numberOfRuns,1));
MISEs(numberOfRuns+3,1) =  std(MISEs(1:numberOfRuns,1));

HLGRs(numberOfRuns+2,1) = mean(HLGRs(1:numberOfRuns,1));
HLGRs(numberOfRuns+3,1) =  std(HLGRs(1:numberOfRuns,1));

trainStageTimes(numberOfRuns+2,1) = mean(trainStageTimes(1:numberOfRuns,1));
trainStageTimes(numberOfRuns+3,1) =  std(trainStageTimes(1:numberOfRuns,1));

marginalsTime(numberOfRuns+2,1) = mean(marginalsTime(1:numberOfRuns,1));
marginalsTime(numberOfRuns+3,1) =  std(marginalsTime(1:numberOfRuns,1));

copulaTime(numberOfRuns+2,1) = mean(copulaTime(1:numberOfRuns,1));
copulaTime(numberOfRuns+3,1) =  std(copulaTime(1:numberOfRuns,1));

testStageTimes(numberOfRuns+2,1) = mean(testStageTimes(1:numberOfRuns,1));
testStageTimes(numberOfRuns+3,1) =  std(testStageTimes(1:numberOfRuns,1));

copulaCuts(end-1,1)=mean(copulaCuts(1:numberOfRuns, 1));
copulaCuts(end,1)=std(copulaCuts(1:numberOfRuns, 1));

for r=1:D+2
    marginalCuts(r,numberOfRuns+2) = mean(marginalCuts(r,1:numberOfRuns));
    marginalCuts(r,numberOfRuns+3) =  std(marginalCuts(r,1:numberOfRuns));
end

if (copulaEnbl==yes)
    display(marginalCuts);
    display(marginalsTime);
end

display(copulaTime);
display(copulaCuts);
display(trainStageTimes);
display(testStageTimes)

if (KLDivEnbl==yes)
    display(KLDs);
    display(MISEs);
    display(HLGRs);
end


fName='results';
save(fName, 'allDensities', 'avgDensities', 'KLDs', 'NMSEs', 'MISEs', 'HLGRs', 'marginalCuts', 'copulaCuts', 'trainStageTimes', 'marginalsTime', 'copulaTime', 'testStageTimes');

delete('allParameters.mat')
delete('allVariables.mat')
delete('allVars*.mat')
delete('best_*.mat')
delete('tree*.mat')

diary off

end

%% end mvdest >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


%% fnc bincut >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function[bestTrees, densMAPs, VkMAPs, bestCoordsMAPs, inputData, marginalCuts, bestCoordsMAPcopula, densMAPcopula, copulaCuts, tMarginal, tCopula, marginalTimes]=bincut(block, allBounds, inputData, copulaEnbl, resmpMarginal, resmpCopula, resmpInterval, jMaxMarginal, MMarginal, jMaxCopula, MCopula, bestCutMarginal, bestCutCopula)

format shortG

fileName='allParameters';
load(fileName);

t0Marginal=tic;

betaa=1;

if (bestCutMarginal==yes)
    MMarginal=1;
end

if (bestCutCopula==yes)
    MCopula=1;
end

N=size(inputData,1);
D=size(inputData,2);

bestTrees = {};

densMAPs = zeros(jMaxMarginal, D);
VkMAPs=zeros(jMaxMarginal, D);
nkMAPs=zeros(jMaxMarginal, D);   
bestCoordsMAPs=zeros(2, jMaxMarginal, D);

marginalCuts=zeros(D,1);
marginalTimes=zeros(D,1);

if (kdeForMarginals)
    
else
    
    if (copulaEnbl==yes)
        
        weights = ones(MMarginal,1);
        
        dataMarginal =zeros(N,3,MMarginal);
        dataMarginal(:,:,1) = [ (1:N)' zeros(N,1) ones(N,1)];
        
        for m=1:MMarginal
            dataMarginal(:,:,m) =  dataMarginal(:,:,1);
        end
        
        dataDistLimits=zeros(2,jMaxMarginal,MMarginal);
        
        dataDistLimits(1,1,1)=1;
        dataDistLimits(2,1,1)=N;
        for m=1:MMarginal
            dataDistLimits(:,:,m)=dataDistLimits(:,:,1);
        end
        
        
        
        % parfor marginal=1:D
        for marginal=1:D
            leaves=ones(1, MMarginal);
            
            left=allBounds(1, marginal);
            right=allBounds(2, marginal);
            
            
            val0=0.5*(left+right);
            val1={1};
            val2=N;
            val3=left;
            val4={-1};
            val5={-1};
            
            
            clear tree_m*
            
            str1='tree_m';
            v = genvarname(str1 , who);
            eval([v ' = 0;'])
            
            treeNames=cell(MMarginal,1);
            
            tree0 = struct('mid', val0, 'label', val1, 'sampleCount', val2, 'leftCoord', val3, 'leftChild', val4, 'rightChild', val5);
            for m=1:MMarginal
                str1='tree_m';
                v = genvarname(str1 , who);
                eval([v ' = tree0;']);
                treeNames{m}=v;
            end
            
        
            j=1;
            
            marginalTime0=tic;
            bounds=zeros(2,1);
            
            dataMarginalD=dataMarginal;
            dataDistLimitsD=dataDistLimits;
            
            VkMAPsD=VkMAPs(:, marginal);
            nkMAPsD=nkMAPs(:, marginal);
            
            bestCoordsMAPsD=bestCoordsMAPs(:, :, marginal);
            
            densMAPsD = densMAPs(:, marginal);
            
            coords=zeros(2,jMaxMarginal, MMarginal);
            cuts=zeros(MMarginal,1);
            score=zeros(MMarginal,1);        % Partition scores of all sample partitions
            maxScore=-Inf;                   % Initial value of maximum score
            bestCoords=coords;
            counter=0;                       % Counts No. of additional steps after a maximum score is reached.
            for m=1:MMarginal
                dataMarginalD(:,2:3,m) = [inputData(:,marginal), ones(N,1)];
            end
            
       
            bounds(1,1)=allBounds(1, marginal);
            bounds(2,1)=allBounds(2, marginal);
            
            nk=zeros(jMaxMarginal,MMarginal);       % Number of data points in each subregion
            Vk=zeros(jMaxMarginal,MMarginal);       % Size of each subregion
            
            coords(1,1,:)=bounds(1,1)-eps;          % First subregion = The whole sample space
            coords(2,1,:)=bounds(2,1)+eps;
            
            for m=1:MMarginal                       % Initialize all the sample partitions
                coords(:,1,m)=coords(:,1,1);        % Coordinates of subregion 1
            end
            
            V0=1;
            V0 = V0 * (bounds(2,1) - bounds(1,1));  % Volume of the whole sample space
            
            for m=1:MMarginal                       % Initialize all the sample partitions
                Vk(1,m)=V0;                         % Subregion 1 is the whole sample space
            end
            for m=1:MMarginal                       % Initialize all the sample partitions
                nk(1,m)=N;                          % All the data is in subregion 1
            end
            
            sjLog=log(zeros((jMaxMarginal-1),MMarginal));
            
            while ( (j < jMaxMarginal) && (counter<10))
                
                j=j+1;
                
                for m=1:MMarginal
 
                    currentTree=eval(treeNames{m});
                     
                    if (cuts(m)==0)
                        newSubregions = ones(1,1);
                    else
                        newSubregions=[cuts(m), j-1];
                    end
                    
                    for k=1:numel(newSubregions)
                        
                        dj=newSubregions(k);
                        
                        begins=dataDistLimitsD(1,dj,m);
                        ends=dataDistLimitsD(2,dj,m);
                        
                        bounds(1,1)= coords(1,dj,m);
                        bounds(2,1)= coords(2,dj,m);
                        
                        nkj=nk(dj,m);
                        
                        if(nkj==0)
                            sjLog(dj, m)=-Inf;
                        else
                            
                            midPoint = 0.5 * (bounds(1,1) + bounds(2,1));
                            
                            nkj2=sum(dataMarginalD(begins:ends,2,m)>= midPoint);
                            nkj1=nkj-nkj2;
                            
                            if (nkj1==0)
                                nkj1=1;
                            end
                            
                            if (nkj2==0)
                                nkj2=1;
                            end
                            
                            sjLog(dj,m) = (nkj)*log(2) + gammaln(nkj1) + gammaln(nkj2) - gammaln(nkj);
                        end
                    end
                    
                    sjl=sjLog(1:(j-1), m);
                    
                    if (bestCutMarginal==no)
                        
                        A= max(max(sjl));
                        
                        sj = exp( sjl - (A+ log( sum(sum(exp(sjl-A))))));
                        
                        sj=sj/sum(sum(sj));
                        sCDF=cumsum(reshape(sj,numel(sj),1));
                        
                        
                        r = rand;                    % Inverse Transform Sampling
                        s=1;
                        
                        while (sCDF(s)<r)
                            s=s+1;                   % Generate a random cut, from the obtained PDF
                        end
                        
                        cutpt=s;                     % Final decision on which subregion to cut
                        
                        cuts(m)=cutpt;
                        
                    else
                        [~, cutpt]= max(sjl);
                    end
                    
                    
                    cuts(m)=cutpt;
                    
                    pt_to_cut=coords(:,cutpt,m);      % Coordinates of the subregion to cut
                    
                    bounds(1,1)= pt_to_cut(1);
                    bounds(2,1)= pt_to_cut(2);
                    
                    midPoint = 0.5 * (bounds(1,1) + bounds(2,1));
                    
                    bounds1 = bounds;                 % boundaries of the first half
                    bounds2 = bounds;                 % boundaries of the second half
                    
                    bounds1(2,1) = midPoint;          % Update coordinates of first half
                    bounds2(1,1) = midPoint;          % Update coordinates of second half
                    
                    newPtn = reshape(bounds2, 2,1);   % Coordinates of newly created subregion
                    
                    coords(:,cutpt,m)=reshape(bounds1, 2, 1);     % Update coordinates of the cut subregion
                    coords(:,j,m)=newPtn;                         % Append coordinates of the new subregion
                    
                    begins=dataDistLimitsD(1,cutpt,m);
                    ends=dataDistLimitsD(2,cutpt,m);
                    rows=dataMarginalD(begins:ends,2,m)>=midPoint;
                    rows=rows.*(begins:ends)';
                    rows=rows(rows~=0);
                    dataMarginalD(rows,3,m)=j;
                    
                    [dataMarginalD(begins:ends,3,m),sortOrder]=sort(dataMarginalD(begins:ends,3,m));
                    dataMarginalD(begins:ends,1:2,m)=dataMarginalD(begins+sortOrder-1,1:2,m);
                    
                    nkj2=size(rows,1);
                    nkj1=(ends-begins+1)-nkj2;
                    
                    Vk(cutpt,m)=Vk(cutpt,m)/2;        % Update volume of the cut subregion
                    Vk(j,m)=Vk(cutpt,m);              % Volume of newly created subregion
                    
                    nk(cutpt,m) = nkj1;               % Update nk for both halves of the cut subregion
                    nk(j,m) = nkj2;
                    
                    if (nkj1==0)
                        dataDistLimitsD(1,cutpt,m)=0;
                        dataDistLimitsD(2,cutpt,m)=0;
                    else
                        dataDistLimitsD(2,cutpt,m)=begins+nkj1-1;
                    end
                    
                    if (nkj2==0)
                        dataDistLimitsD(1,j,m)=0;
                        dataDistLimitsD(2,j,m)=0;
                    else
                        dataDistLimitsD(1,j,m)=begins+nkj1;
                        dataDistLimitsD(2,j,m)=ends;
                    end
                    
                    nextCut=cutpt;
                    nextCutNodeNumber=leaves(nextCut, m);
                    n1=nkj1;
                    currentTree=binTree(currentTree, nextCutNodeNumber, n1);
                    leaves(nextCut, m)=size(currentTree,2)-1;
                    leaves(end-(m>1)+1, m)=size(currentTree,2);
                    
                    eval([treeNames{m} ' = currentTree;']);
                    %  parsave(treeName, tree);
                    
                    Alphas=max( min( nk(1:j,m)/200 , (1/2)* ones(j,1) ), (1/10)*ones(j,1) );
                    
                    score(m) = -betaa*j + sum(gammaln(nk(1:j,m)+Alphas(1:j)))...
                        - ( gammaln(sum(nk(1:j,m)+Alphas(1:j))))- sum( gammaln(Alphas(1:j)))...
                        + ( gammaln(sum(Alphas(1:j)))) - sum( nk(1:j,m).*log(Vk(1:j,m)) );
                 
                end
                
                
                %% Resampling
                if (resmpMarginal==yes)
                    gamma=0.5;
                    if (mod(j,resmpInterval)==0)
                        resamp(weights, gamma);
                    end
                end
                
                %%  Save the best partitions so far
                
                if (max(score)>maxScore)             % If a max score is obtained...
                    maxScore=max(score);
                    jBest=j;                         % Store No. of cuts for best score
                    counter=0;                       % Reset the counter (and go 10 more steps)
                    bestCoords=coords(:,1:j,:);      % Coordinates of the best partition
                    nkBest=nk(1:j,:);                % nk for the best partitioning scheme
                    VkBest=Vk(1:j,:);                % Vk for the best partitioning scheme
                    bestDist=dataMarginalD;          % data distribution for best partitioning scheme
                    bestScores=score;
                    
                    mBest = find(bestScores==max(bestScores));
                    mBest = mBest(1);   % tie-breaker

                    bestScoreTree=eval(treeNames{mBest});
                    
                    %%
                    
                    bestTreeName=['best_d' int2str(marginal)];
%                   parsave(bestTreeName, bestScoreTree);
                    bestTrees{marginal}=bestTreeName;
                else
                    counter=counter+1;
                end
            end
            
            parsave(bestTreeName, bestScoreTree);

            str1='allVars_d';
            str2=num2str(marginal);
            fName=[str1 str2];
            save(fName, 'jBest', 'cuts', 'coords', 'score');
            
            marginalCuts(marginal)=jBest;
            
            
            %% MAP estimate 
            
            mMAPs = find(bestScores==max(bestScores));
            mMAP = mMAPs(1);
            nkMAP = nkBest(:, mMAP);
            
            VkMAPsD(1:marginalCuts(marginal))= VkBest(:, mMAP);
            
            nkMAPsD(1:marginalCuts(marginal))= nkBest(:, mMAP); 
            
            bestDistMAP = bestDist(:, :, mMAP) ;
            bestCoordsMAP= bestCoords(:,:,mMAP);
            bestCoordsMAPsD(:,1:marginalCuts(marginal)) = bestCoordsMAP;
            
            for p=1:marginalCuts(marginal)
                densMAPsD(p)=nkMAPsD(p)/(N*VkMAPsD(p));
            end
            
            
            %% Calculate CDF
            
            if(D>1)
                dataMarginalD(:,:,mMAP)=bestDistMAP;
                
                prevPtn = 1;
                cdfCum = 0;
                
                for i=1:N
                    z=dataMarginalD(i,2,mMAP);
                    ptnNum=dataMarginalD(i,3,mMAP);
                    
                    if (ptnNum ~= prevPtn)
                        cdfCum = cdfCum + VkMAPsD(prevPtn)*densMAPsD(prevPtn);
                    end
                    ptnBegin=bestCoordsMAPsD(1,ptnNum);
                    cdf = cdfCum + (z - ptnBegin)*densMAPsD(ptnNum);
                    dataMarginalD(i,2,mMAP) =cdf;
                    prevPtn = ptnNum;
                end
                
                [dataMarginalD(:,1,mMAP),sortOrder]=sort(dataMarginalD(:,1,mMAP));
                dataMarginalD(:,2,mMAP)=dataMarginalD(sortOrder,2,mMAP);
                
                
                for m=1:MMarginal
                    dataMarginalD(:,1,m)=(1:N)';
                end
                inputData(:,marginal) = dataMarginalD(:,2,mMAP);
                
                densMAPs(:, marginal)=densMAPsD;
                VkMAPs(:, marginal)=VkMAPsD;
                
                nkMAPs(:, marginal)=nkMAPsD;
                bestCoordsMAPs(:, :, marginal)=bestCoordsMAPsD;
                
                marginalTimes(marginal)=toc(marginalTime0);
                
            end
            
        end
        
        fName='allVars_best';
        save(fName, 'nkMAPs', 'VkMAPs', 'bestCoordsMAPs', 'marginalCuts');
        
        save('allTrees', 'tree_m*');
    end
    
end

tMarginal = toc(t0Marginal);
t0Copula= tic;



%% Copula estimation (if copula enabled) / Direct cuts on d-dimensioan space (if copula disabled)

densMAPcopula = zeros(jMaxCopula, 1);

cuts=zeros(MCopula,1);

score=zeros(MCopula,1);         % Partition scores of all sample partitions
maxScore=-Inf;                  % Initial value of maximum score

coords=zeros((2*D),jMaxCopula, MCopula);   % Stores coordinates of the subregions
bestCoords=coords;

counter=0;                      % Counts No. of additional steps after a maximum score is reached.

dataDist=zeros(N,2,MCopula);
dataDist(:,1,1)=1:N;
dataDist(:,2,1)=ones(N,1);
sjLog=log(zeros(D,(jMaxCopula-1),MCopula));

for m=1:MCopula
    dataDist(:,1:2,m)=dataDist(:,1:2,1);
end

dataDistLimits=zeros(2,jMaxCopula,MCopula);
dataDistLimits(1,1,1)=1;
dataDistLimits(2,1,1)=N;

for m=1:MCopula
    dataDistLimits(:,:,m)=dataDistLimits(:,:,1);
end

bounds=zeros(2,D);                          % Bounds (min, max) of data in each of the dimensions

for d=1:D
    bounds(1,d)=min(inputData(:,d));        % Bound sample space to min and max of data
    bounds(2,d)=max(inputData(:,d));
end

nk=repmat(-1,jMaxCopula,MCopula);           % Number of data points in each subregion

Vk=repmat(-1,jMaxCopula,MCopula);           % Size of each subregion

for d=1:D
    coords(2*d-1,1,:)=bounds(1,d)-eps;      % First subregion = The whole sample space
    coords(2*d,1,:)=bounds(2,d)+eps;
end
for m=1:MCopula                             % Initialize all the sample partitions
    coords(:,1,m)=coords(:,1,1);            % Coordinates of subregion 1
end
V0=1;
for d=1:D
    V0 = V0 * (bounds(2,d) - bounds(1,d));  % Volume of the whole sample space
end
for m=1:MCopula                   % Initialize all the sample partitions
    Vk(1,m)=V0;                   % Subregion 1 is the whole sample space
end
for m=1:MCopula                   % Initialize all the sample partitions
    nk(1,m)=N;                    % All the data is in subregion 1
end

weights = ones(MCopula,1);

j=1;

while ( (j < jMaxCopula) && (counter<10))
    
    j=j+1;
    
    for m=1:MCopula
        
        if (cuts(m)==0)
            newSubregions = [1];
        else
            newSubregions=[cuts(m), j-1];
        end
        
        
        for k=1:numel(newSubregions)
            
            dj=newSubregions(k);
            
            begins=dataDistLimits(1,dj,m);
            ends=dataDistLimits(2,dj,m);
            
            for d=1:D
                bounds(1,d)= coords(2*d-1,dj,m);
                bounds(2,d)= coords(2*d,dj,m);
            end
            
            nkj=nk(dj,m);
            
            if(nkj==0)
                sjLog(:,dj, m)=-Inf;
            else
                
                for dim=1:D
                    
                    midPoint = 0.5 * (bounds(1,dim) + bounds(2,dim));
                    
                    nkj2=sum(inputData(dataDist(begins:ends,1,m), dim)>= midPoint);
                    
                    nkj1=nkj-nkj2;
                  
                 
                    if (nkj1==0)
                        nkj1=1;
                    end
                    
                    if (nkj2==0)
                        nkj2=1;
                    end
                    
                    sjLog(dim,dj, m) = (nkj)*log(2) + gammaln(nkj1) + gammaln(nkj2) - gammaln(nkj);
                end
            end
        end
        
        sjl=sjLog(:, 1:(j-1), m);
        A= max(max(sjl));
        sj = exp( sjl - (A+ log( sum(sum(exp(sjl-A))))));
        
     
        sj=sj/sum(sum(sj));
        sCDF=cumsum(reshape(sj,numel(sj),1));
        
        r = rand;                     % Inverse Transform Sampling
        s=1;
        
        while (sCDF(s)<r)
            s=s+1;                    % Generate a random cut, from the obtained PDF
        end
        
        cutpt=ceil(s/D);              % Final decision on which subregion to cut
        cutdim = 1 + mod((s-1),D);    % Final decision on which dimension to cut
        cuts(m)=cutpt;
        
        pt_to_cut=coords(:,cutpt,m);  % Coordinates of the subregion to cut
        
        for d=1:D
            bounds(1,d)= pt_to_cut(2*d-1);
            bounds(2,d)= pt_to_cut(2*d);
        end
        
        midPoint = 0.5 * (bounds(1,cutdim) + bounds(2,cutdim));
        
        bounds1 = bounds;                 % boundaries of the first half
        bounds2 = bounds;                 % boundaries of the second half
        
        bounds1(2,cutdim) = midPoint;     % Update coordinates of first half
        bounds2(1,cutdim) = midPoint;     % Update coordinates of second half
        
        newPtn = reshape(bounds2, 2*D,1);             % Coordinates of newly created subregion
        
        coords(:,cutpt,m)=reshape(bounds1, 2*D, 1);   % Update coordinates of the cut subregion
        coords(:,j,m)=newPtn;                         % Append coordinates of the new subregion
        
        
        begins=dataDistLimits(1,cutpt,m);
        ends=dataDistLimits(2,cutpt,m);
        
        rows=inputData(dataDist(begins:ends,1,m),cutdim)>= midPoint;
        rows=rows.*(begins:ends)';
        rows=rows(rows~=0);
        dataDist(rows,2,m)=j;
        
        [dataDist(begins:ends,2,m),sortOrder]=sort(dataDist(begins:ends,2,m));
        dataDist(begins:ends,1,m)=dataDist(begins+sortOrder-1,1,m);
        
        nkj2=size(rows,1);
        nkj1=(ends-begins+1)-nkj2;
        
        Vk(cutpt,m)=Vk(cutpt,m)/2;        % Update volume of the cut subregion
        Vk(j,m)=Vk(cutpt,m);              % Volume of newly created subregion
        
        nk(cutpt,m) = nkj1;               % Update nk for both halves of the cut subregion
        nk(j,m) = nkj2;
        
        dataDistLimits(2,cutpt,m)=nkj1+begins-1;
        dataDistLimits(1,j,m)=nkj1+begins;
        dataDistLimits(2,j,m)=ends;
        
        Alphas=max( min( nk(1:j,m)/200 , (1/2)* ones(j,1) ), (1/10)*ones(j,1) );
        
        score(m) = -betaa*j + sum(gammaln(nk(1:j,m)+Alphas(1:j)))...
            - ( gammaln(sum(nk(1:j,m)+Alphas(1:j))))- sum( gammaln(Alphas(1:j)))...
            + ( gammaln(sum(Alphas(1:j)))) - sum( nk(1:j,m).*log(Vk(1:j,m)) );
        
    end
    
    %% Resampling
    if (resmpCopula==yes)
        gamma=0.5;
        if (mod(j,resmpInterval)==0)
            resamp(weights, gamma);
        end
    end
    
    %%  Save thte best partitions so far
    
    if (max(score)>maxScore)                % If a max score is obtained...
        maxScore=max(score);
        jBest=j;                            % Store No. of cuts for best score
        counter=0;                          % Reset the counter (and go 10 more steps)
        bestCoords=coords(:,1:j,:);         % Coordinates of the best partition
        nkBestCoords=nk(1:j,:);             % nk for the best partitioning scheme
        VkBestCoords=Vk(1:j,:);             % Vk for the best partitioning scheme
        best_scores=score;
    else
        counter=counter+1;
    end
    
end

copulaCuts=jBest;

% Apply weights to the sample partitions (based on Lambda)

w_xt=ones(MCopula,1);

pi_xt= best_scores + log(w_xt);

mMAPs = find(pi_xt==max(pi_xt));

mMAP = mMAPs(1);

nkMAP = nkBestCoords(:, mMAP);
VkMAP = VkBestCoords(:, mMAP);
bestCoordsMAPcopula = bestCoords(:,1:jBest,mMAP);

for p=1:jBest
    densMAPcopula(p)=nkMAP(p)/(N*VkMAP(p));
end

tCopula = toc(t0Copula);

end

%% end bincut >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc bincut_progressive >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function[bestTrees, densMAPs, VkMAPs, bestCoordsMAPs, marginalCuts, bestCoordsMAPcopula, densMAPcopula, copulaCuts, tMarginal, tCopula]=...
    bincut_progressive(block, allBounds, inputData, copulaEnbl, resmpMarginal, resmpCopula, resmpInterval, jMaxMarginal, MMarginal, jMaxCopula, MCopula, bestCutMarginal, bestCutCopula)

format shortG

fileName='allParameters';
load(fileName);

t0Marginal=tic;
betaa=1;

if (bestCutMarginal==yes)
    MMarginal=1;
end

if (bestCutCopula==yes)
    MCopula=1;
end

N=size(inputData,1);
D=size(inputData,2);


densMAPs = zeros(jMaxMarginal, D);

marginalTimes=zeros(D,1);

if (copulaEnbl==yes)
    
    dataMarginal =zeros(N,3,MMarginal);
    dataMarginal(:,:,1) = [ (1:N)' zeros(N,1) ones(N,1)];
    
    for m=1:MMarginal
        dataMarginal(:,:,m) =  dataMarginal(:,:,1);
    end
    
    dataDistLimits=zeros(2,jMaxMarginal,MMarginal);
    
    for m=1:MMarginal
        dataDistLimits(:,:,m)=dataDistLimits(:,:,1);
    end

    load('allVars_best');

    for marginal=1:D
        
        bestTreeName=['best_d' int2str(marginal)];
        load(bestTreeName)
      
        tree0=x;

        str1='allVars_d';
        str2=num2str(marginal);
        fName=[str1 str2];
        load(fName);
        
        coords=zeros(2, jMaxMarginal, MMarginal);
        for m=1:MMarginal
            coords(:,:,m)=bestCoordsMAPs(:,:, marginal);
        end
        
        dataMarginalD=dataMarginal;
        dataDistLimitsD=dataDistLimits;     
        
        newCount=zeros(marginalCuts(marginal), 1);
       
        
assignedSubs = findSubregion(tree0, inputData(:,marginal));        
        
        for n=1:N
            newCount(assignedSubs(n)) = newCount(assignedSubs(n)) + 1;
        end
   
        
        for m=1:MMarginal
            dataMarginalD(:,2:3,m) = [inputData(:,marginal), assignedSubs];
            dataMarginalD(:,:,m) = sortrows(dataMarginalD(:,:,m), 3);
        end
        
        m=1;
        
        index=cumsum(newCount);
        
        
        both=[(1:marginalCuts(marginal))' newCount];
               
        
        dataDistLimitsD(1,1,m)=(newCount(1)>0);
        dataDistLimitsD(2,1,m)=newCount(1);
        
        
        for j0=2:marginalCuts(marginal)
            if (both(j0,2)>0)
                dataDistLimitsD(1,j0,m)=index(j0-1)+1;
                dataDistLimitsD(2,j0,m)=index(j0);
            end
        end
        
        
        for m=1:MMarginal   
            dataDistLimitsD(:,:,m) = dataDistLimitsD(:,:,1);
        end
        
        VkMAPsD=VkMAPs(:, marginal);
        nkMAPsD=nkMAPs(:, marginal);
        
        Vk=VkMAPs(:,marginal);

        leaves=1;       
       
        lChild=cell2mat({tree0.leftChild});
        leaves=find(lChild==-1);
        labs=cell2mat({tree0(leaves).label});
        merged=[labs' leaves'];
        merged=sortrows(merged, 1);
        leaves=merged(:,2);

        
        
        nkMAPsNew=zeros(size(nkMAPs));
        nkMAPsNew(1:marginalCuts(marginal), marginal) = newCount;
        total=nkMAPsNew + nkMAPs;
        
        
        %% If we decide to pick the subregions with largest increase:
        
        y=[(1:marginalCuts(marginal))' nkMAPs(1:marginalCuts(marginal), marginal) total(1:marginalCuts(marginal), marginal)];
        
        delta=y(:,3)./y(:,2);
        y(:,4)=delta;
        y=y( y(:,4)~=Inf ,:);
        y=y(~isnan(y(:,4)),:);
        ySorted=sortrows(y, 4);
        ySorted=flipud(ySorted);
        
        j=marginalCuts(marginal);
        
        nkMAPs(:, marginal)=nkMAPsNew(:, marginal);
        
        nk=nkMAPs(:, marginal);

        nk=zeros(jMaxMarginal, MMarginal);
        Vk=zeros(jMaxMarginal, MMarginal);
        
        for m=1:MMarginal
            nk(:, m)=nkMAPs(:, marginal);
            Vk(:, m)=VkMAPs(:, marginal);
        end
        
        marginalTime0=tic;

        VkMAPsD=VkMAPs(:, marginal);
        bestCoordsMAPsD=bestCoordsMAPs(:, :, marginal);
        densMAPsD = densMAPs(:, marginal);
        cuts=zeros(MMarginal,1);
        score=zeros(MMarginal,1);        % Partition scores of all sample partitions
        maxScore=-Inf;                   % Initial value of maximum score
        
        counter=0;                       % Counts No. of additional steps after a maximum score is reached.

        sjLog=log(zeros((jMaxMarginal-1),MMarginal));
       
        x=tree0;
               
        treeNames=cell(MMarginal,1);

        clear tree_m*
        
        str1='tree_m';
        v = genvarname(str1 , who);
        eval([v ' = 0;'])
            
        for m=1:MMarginal
            v = genvarname(str1 , who);
            eval([v ' = x;']);
            treeNames{m}=v;
        end   
        
        jm=jBest+1;
        j0=jBest;
        while ( (j < jMaxMarginal) && (counter<10))
            
            j=j+1;
            
            for m=1:MMarginal
              
                currentTree=eval(treeNames{m});

                if (cuts(m)==0)
                    newSubregions = (1:j-1)';
                else
                    newSubregions=[cuts(m), j-1];
                end
                
                
                for k=1:numel(newSubregions)
                    
                    dj=newSubregions(k);
                    
                    begins=dataDistLimitsD(1,dj,m);
                    ends=dataDistLimitsD(2,dj,m);

                    bounds(1,1)= coords(1,dj,m);
                    bounds(2,1)= coords(2,dj,m);
                    
                    nkj=nk(dj,m);
                    
                    if(nkj==0)
                        sjLog(dj, m)=-Inf;
                    else
                        
                        midPoint = 0.5 * (bounds(1,1) + bounds(2,1));
                        
                        nkj2=sum(dataMarginalD(begins:ends,2,m)>= midPoint);
                        nkj1=nkj-nkj2;
                        
                        if (nkj1==0)
                            nkj1=1;
                        end
                        
                        if (nkj2==0)
                            nkj2=1;
                        end
                        
                        sjLog(dj,m) = (nkj)*log(2) + gammaln(nkj1) + gammaln(nkj2) - gammaln(nkj);
                    end
                end
                
                sjl=sjLog(1:(j-1), m);
                
                if (bestCutMarginal==no)
                    A= max(max(sjl));
                    
                    sj = exp( sjl - (A+ log( sum(sum(exp(sjl-A))))));
                                       
                    sj=sj/sum(sum(sj));
                    sCDF=cumsum(reshape(sj,numel(sj),1));
                    
                    r = rand;                    % Inverse Transform Sampling
                    s=1;
                    
                    while (sCDF(s)<r)
                        s=s+1;                   % Generate a random cut, from the obtained PDF
                    end
                    
                    cutpt=s;                     % Final decision on which subregion to cut

                    cuts(m)=cutpt;
                    
                else
                    [~, cutpt]= max(sjl);
                end

                cuts(m)=cutpt;
                
                
                pt_to_cut=coords(:,cutpt,m);      % Coordinates of the subregion to cut
                
                bounds(1,1)= pt_to_cut(1);
                bounds(2,1)= pt_to_cut(2);
                
                midPoint = 0.5 * (bounds(1,1) + bounds(2,1));
                
                bounds1 = bounds;                 % boundaries of the first half
                bounds2 = bounds;                 % boundaries of the second half
                
                bounds1(2,1) = midPoint;          % Update coordinates of first half
                bounds2(1,1) = midPoint;          % Update coordinates of second half
                
                newPtn = reshape(bounds2, 2,1);   % Coordinates of newly created subregion
                
                coords(:,cutpt,m)=reshape(bounds1, 2, 1);     % Update coordinates of the cut subregion
                coords(:,j,m)=newPtn;                         % Append coordinates of the new subregion
                
                begins=dataDistLimitsD(1,cutpt,m);
                ends=dataDistLimitsD(2,cutpt,m);
                
                rows=dataMarginalD(begins:ends,2,m)>=midPoint;
                rows=rows.*(begins:ends)';
                rows=rows(rows~=0);
                dataMarginalD(rows,3,m)=j;
                
                [dataMarginalD(begins:ends,3,m),sortOrder]=sort(dataMarginalD(begins:ends,3,m));
                dataMarginalD(begins:ends,1:2,m)=dataMarginalD(begins+sortOrder-1,1:2,m);
                
                nkj2=size(rows,1);
                nkj1=(ends-begins+1)-nkj2;
                
                Vk(cutpt,m)=Vk(cutpt,m)/2;        % Update volume of the cut subregion
                Vk(j,m)=Vk(cutpt,m);              % Volume of newly created subregion
                
                nk(cutpt,m) = nkj1;               % Update nk for both halves of the cut subregion
                nk(j,m) = nkj2;
                
                if (nkj1==0)
                    dataDistLimitsD(1,cutpt,m)=0;
                    dataDistLimitsD(2,cutpt,m)=0;
                else
                    dataDistLimitsD(2,cutpt,m)=begins+nkj1-1;
                end
                
                if (nkj2==0)
                    dataDistLimitsD(1,j,m)=0;
                    dataDistLimitsD(2,j,m)=0;
                else
                    dataDistLimitsD(1,j,m)=begins+nkj1;
                    dataDistLimitsD(2,j,m)=ends;
                end
                

                lChild=cell2mat({currentTree.leftChild});
                leaves=find(lChild==-1);
                labs=cell2mat({currentTree(leaves).label});
                merged=[labs' leaves'];
                merged=sortrows(merged, 1);
                leaves=merged(:,2);

                nextCut=cutpt;
                nextCutNodeNumber=leaves(nextCut);
                n1=nkj1;
                currentTree=binTree(currentTree, nextCutNodeNumber, n1);
                leaves(nextCut)=size(currentTree,2)-1;
                leaves(end+1)=size(currentTree,2);
                
                eval([treeNames{m} ' = currentTree;']);
 
                Alphas=max( min( nk(1:j,m)/200 , (1/2)* ones(j,1) ), (1/10)*ones(j,1) );
                
                score(m) = -betaa*j + sum(gammaln(nk(1:j,m)+Alphas(1:j)))...
                    - ( gammaln(sum(nk(1:j,m)+Alphas(1:j))))- sum( gammaln(Alphas(1:j)))...
                    + ( gammaln(sum(Alphas(1:j)))) - sum( nk(1:j,m).*log(Vk(1:j,m)) );
                
            end
            
            
            %% Resampling
            if (resmpMarginal==yes)
                gamma=0.5;
                if (mod(j,resmpInterval)==0)
                    resamp(weights, gamma);
                end
            end
            
            %%  Save the best partitions so far
            if (max(score)>maxScore)             % If a max score is obtained...
                maxScore=max(score);
                jBest=j;                         % Store No. of cuts for best score
                counter=0;                       % Reset the counter (and go 10 more steps)
                bestCoords=coords(:,1:j,:);      % Coordinates of the best partition
                nkBest=nk(1:j,:);                % nk for the best partitioning scheme
                VkBest=Vk(1:j,:);                % Vk for the best partitioning scheme
                bestDist=dataMarginalD;          % data distribution for best partitioning scheme
                bestScores=score;

                
                
                mBest = find(bestScores==max(bestScores));
                mBest = mBest(1);   % tie-breaker
                              
                bestScoreTree=eval(treeNames{mBest});
        
                bestTreeName=['best_d' int2str(marginal)];
                bestTrees{marginal}=bestTreeName;
            else
                counter=counter+1;
            end
end
        
        parsave(bestTreeName, bestScoreTree);
        

        str1='allVars_d';
        str2=num2str(marginal);
        fName=[str1 str2];
        save(fName, 'jBest', 'cuts', 'coords', 'score');
        
        marginalCuts(marginal)=jBest;
        
       
        %% MAP estimate 
        
        mMAPs = find(bestScores==max(bestScores));
        mMAP = mMAPs(1);

        nkMAP = nkBest(:, mMAP);
        
        VkMAPsD(1:marginalCuts(marginal))= VkBest(:, mMAP);
        
        nkMAPsD(1:marginalCuts(marginal))= nkBest(:, mMAP); %%%% 06_12
        
        bestDistMAP = bestDist(:, :, mMAP) ;
        bestCoordsMAP= bestCoords(:,:,mMAP);
        bestCoordsMAPsD(:,1:marginalCuts(marginal)) = bestCoordsMAP;
  
       
        for p=1:marginalCuts(marginal)
            densMAPsD(p)=nkMAPsD(p)/(N*VkMAPsD(p));
        end
        
        marginalCutsTemp(marginal)=marginalCuts(marginal);
        
        %% Calculate CDF
        
        if(D>1)
            dataMarginalD(:,:,mMAP)=bestDistMAP;
            
            dataMarginalD(:,:,mMAP)=sortrows(dataMarginalD(:,:,mMAP), 2);
            
            prevPtn = 1;
            cdfCum = 0;
            
            for i=1:N
                z=dataMarginalD(i,2,mMAP);
                ptnNum=dataMarginalD(i,3,mMAP);
                
                if (ptnNum ~= prevPtn)
                    cdfCum = cdfCum + VkMAPsD(prevPtn)*densMAPsD(prevPtn);
                end
                ptnBegin=bestCoordsMAPsD(1,ptnNum);
                cdf = cdfCum + (z - ptnBegin)*densMAPsD(ptnNum);
                dataMarginalD(i,2,mMAP) =cdf;
                prevPtn = ptnNum;
            end
            
            [dataMarginalD(:,1,mMAP),sortOrder]=sort(dataMarginalD(:,1,mMAP));
            dataMarginalD(:,2,mMAP)=dataMarginalD(sortOrder,2,mMAP);
            
            
            for m=1:MMarginal
                dataMarginalD(:,1,m)=(1:N)';
            end
            inputData(:,marginal) = dataMarginalD(:,2,mMAP);
            
            densMAPs(:, marginal)=densMAPsD;
            VkMAPs(:, marginal)=VkMAPsD;
            
            nkMAPs(:, marginal)=nkMAPsD;
            bestCoordsMAPs(:, :, marginal)=bestCoordsMAPsD;
            
            marginalTimes(marginal)=toc(marginalTime0);
            
        end
        
    end
    
    marginalCuts=marginalCutsTemp;
    
    fName='allVars_best';
    save(fName, 'nkMAPs', 'VkMAPs', 'bestCoordsMAPs', 'marginalCuts');
end

tMarginal = toc(t0Marginal);
t0Copula= tic;


%% Copula estimation (if copula enabled) / Direct cuts on d-dimensioan space (if copula disabled)

densMAPcopula = zeros(jMaxCopula, 1);

cuts=zeros(MCopula,1);

score=zeros(MCopula,1);         % Partition scores of all sample partitions
maxScore=-Inf;                  % Initial value of maximum score

coords=zeros((2*D),jMaxCopula, MCopula);   % Stores coordinates of the subregions
bestCoords=coords;

counter=0;                      % Counts No. of additional steps after a maximum score is reached.

dataDist=zeros(N,2,MCopula);
dataDist(:,1,1)=1:N;
dataDist(:,2,1)=ones(N,1);
sjLog=log(zeros(D,(jMaxCopula-1),MCopula));

for m=1:MCopula
    dataDist(:,1:2,m)=dataDist(:,1:2,1);
end

dataDistLimits=zeros(2,jMaxCopula,MCopula);
dataDistLimits(1,1,1)=1;
dataDistLimits(2,1,1)=N;

for m=1:MCopula
    dataDistLimits(:,:,m)=dataDistLimits(:,:,1);
end

bounds=zeros(2,D);                          % Bounds (min, max) of data in each of the dimensions

for d=1:D
    bounds(1,d)=min(inputData(:,d));        % Bound sample space to min and max of data
    bounds(2,d)=max(inputData(:,d));
end

nk=repmat(-1,jMaxCopula,MCopula);           % Number of data points in each subregion

Vk=repmat(-1,jMaxCopula,MCopula);           % Size of each subregion

for d=1:D
    coords(2*d-1,1,:)=bounds(1,d)-eps;      % First subregion = The whole sample space
    coords(2*d,1,:)=bounds(2,d)+eps;
end
for m=1:MCopula                             % Initialize all the sample partitions
    coords(:,1,m)=coords(:,1,1);            % Coordinates of subregion 1
end
V0=1;
for d=1:D
    V0 = V0 * (bounds(2,d) - bounds(1,d));  % Volume of the whole sample space
end
for m=1:MCopula                   % Initialize all the sample partitions
    Vk(1,m)=V0;                   % Subregion 1 is the whole sample space
end
for m=1:MCopula                   % Initialize all the sample partitions
    nk(1,m)=N;                    % All the data is in subregion 1
end

weights = ones(MCopula,1);

j=1;

while ( (j < jMaxCopula) && (counter<10))
    
    j=j+1;
    
    for m=1:MCopula
        
        if (cuts(m)==0)
            newSubregions = [1];
        else
            newSubregions=[cuts(m), j-1];
        end
        
        
        for k=1:numel(newSubregions)
            
            dj=newSubregions(k);
            
            begins=dataDistLimits(1,dj,m);
            ends=dataDistLimits(2,dj,m);
            
            for d=1:D
                bounds(1,d)= coords(2*d-1,dj,m);
                bounds(2,d)= coords(2*d,dj,m);
            end
            
            nkj=nk(dj,m);
            
            if(nkj==0)
                sjLog(:,dj, m)=-Inf;
            else
                
                for dim=1:D
                    
                    midPoint = 0.5 * (bounds(1,dim) + bounds(2,dim));
                    
                    nkj2=sum(inputData(dataDist(begins:ends,1,m), dim)>= midPoint);
                    
                    nkj1=nkj-nkj2;
                    
                    if (nkj1==0)
                        nkj1=1;
                    end
                    
                    if (nkj2==0)
                        nkj2=1;
                    end
                    
                    sjLog(dim,dj, m) = (nkj)*log(2) + gammaln(nkj1) + gammaln(nkj2) - gammaln(nkj);
                end
            end
        end
        
        sjl=sjLog(:, 1:(j-1), m);
        A= max(max(sjl));
        sj = exp( sjl - (A+ log( sum(sum(exp(sjl-A))))));
        
        sj=sj/sum(sum(sj));
        sCDF=cumsum(reshape(sj,numel(sj),1));
        
        r = rand;                     % Inverse Transform Sampling
        s=1;
        
        while (sCDF(s)<r)
            s=s+1;                    % Generate a random cut, from the obtained PDF
        end
        
        cutpt=ceil(s/D);              % Final decision on which subregion to cut
        cutdim = 1 + mod((s-1),D);    % Final decision on which dimension to cut
        cuts(m)=cutpt;
        
        pt_to_cut=coords(:,cutpt,m);  % Coordinates of the subregion to cut
        
        for d=1:D
            bounds(1,d)= pt_to_cut(2*d-1);
            bounds(2,d)= pt_to_cut(2*d);
        end
        
        midPoint = 0.5 * (bounds(1,cutdim) + bounds(2,cutdim));
        
        bounds1 = bounds;                 % boundaries of the first half
        bounds2 = bounds;                 % boundaries of the second half
        
        bounds1(2,cutdim) = midPoint;     % Update coordinates of first half
        bounds2(1,cutdim) = midPoint;     % Update coordinates of second half
        
        newPtn = reshape(bounds2, 2*D,1);             % Coordinates of newly created subregion
        
        coords(:,cutpt,m)=reshape(bounds1, 2*D, 1);   % Update coordinates of the cut subregion
        coords(:,j,m)=newPtn;                         % Append coordinates of the new subregion
        
        
        begins=dataDistLimits(1,cutpt,m);
        ends=dataDistLimits(2,cutpt,m);
        
        rows=inputData(dataDist(begins:ends,1,m),cutdim)>= midPoint;
        rows=rows.*(begins:ends)';
        rows=rows(rows~=0);
        dataDist(rows,2,m)=j;
        
        [dataDist(begins:ends,2,m),sortOrder]=sort(dataDist(begins:ends,2,m));
        dataDist(begins:ends,1,m)=dataDist(begins+sortOrder-1,1,m);
        
        nkj2=size(rows,1);
        nkj1=(ends-begins+1)-nkj2;
        
        Vk(cutpt,m)=Vk(cutpt,m)/2;        % Update volume of the cut subregion
        Vk(j,m)=Vk(cutpt,m);              % Volume of newly created subregion
        
        nk(cutpt,m) = nkj1;               % Update nk for both halves of the cut subregion
        nk(j,m) = nkj2;
        
        dataDistLimits(2,cutpt,m)=nkj1+begins-1;
        dataDistLimits(1,j,m)=nkj1+begins;
        dataDistLimits(2,j,m)=ends;
        
        Alphas=max( min( nk(1:j,m)/200 , (1/2)* ones(j,1) ), (1/10)*ones(j,1) );
        
        score(m) = -betaa*j + sum(gammaln(nk(1:j,m)+Alphas(1:j)))...
            - ( gammaln(sum(nk(1:j,m)+Alphas(1:j))))- sum( gammaln(Alphas(1:j)))...
            + ( gammaln(sum(Alphas(1:j)))) - sum( nk(1:j,m).*log(Vk(1:j,m)) );
        
    end
    
    %% Resampling
    if (resmpCopula==yes)
        gamma=0.5;
        if (mod(j,resmpInterval)==0)
            resamp(weights, gamma);
        end
    end
    
    %%  Save thte best partitions so far
    
    if (max(score)>maxScore)                % If a max score is obtained...
        maxScore=max(score);
        jBest=j;                            % Store No. of cuts for best score
        counter=0;                          % Reset the counter (and go 10 more steps)
        bestCoords=coords(:,1:j,:);         % Coordinates of the best partition
        nkBestCoords=nk(1:j,:);             % nk for the best partitioning scheme
        VkBestCoords=Vk(1:j,:);             % Vk for the best partitioning scheme
        best_scores=score;
    else
        counter=counter+1;
    end
    
end

copulaCuts=jBest;

% Apply weights to the sample partitions (based on Lambda)

w_xt=ones(MCopula,1);

pi_xt= best_scores + log(w_xt);

mMAPs = find(pi_xt==max(pi_xt));

mMAP = mMAPs(1);

nkMAP = nkBestCoords(:, mMAP);
VkMAP = VkBestCoords(:, mMAP);
bestCoordsMAPcopula = bestCoords(:,1:jBest,mMAP);

for p=1:jBest
    densMAPcopula(p)=nkMAP(p)/(N*VkMAP(p));
end

tCopula = toc(t0Copula);

end

%% end bincut_progressive >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc plots >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function plots(trainData, testData, actualDensities, marginalCdfs, margCuts, copCuts, densMAPs, VkMAPs, bestCoordsMAPs, copulaCoords, subRegDensCopula)

disp('Generating plots...')

fileName='allParameters';
load(fileName);

D=size(trainData, 2);

x=(copulaCoords(1,:))';
x=[x;max(copulaCoords(2,:))];

y=subRegDensCopula(1:copCuts);
y=[y;0];

xy=sortrows([x y], 1);
x=xy(:,1);
y=xy(:,2);

if (D==1)
    
    figure
    
    ax1=gca
    stairs(x, y);
    
    p=ax1.XLim;
    p1=p(1);
    p2=p(2);
    hold on
    stairs([p1;x;p2], [0;y;0])
    
    hold on
    xy=[testData actualDensities];
    xy=sortrows(xy, 1);
    plot(xy(:,1), xy(:,2), '.-');
    
elseif(D==2)
    
    % Plot Marginals
    
    cdfs=zeros(max(margCuts), 2);
    
    ver = version('-release');
    v=str2double(ver(1:4));
    
    if (copulaEnbl && v>2015)
        figure
        plot(trainData(:,1), trainData(:,2), '.')
        title('Original Data')
        xlabel('X1')
        ylabel('X2')

        figure
        title('Marginal PDF and CDF')
        
        for i=1:2
            
            % Plot the marginal PDFs
            g=densMAPs(1:margCuts(i),i);
            
            x=bestCoordsMAPs(:,1:margCuts(i),i);
            x=x';
            x=x(:,1);
            
            both=[x g];
            both=sortrows(both, 1);
            
            both=[both ; both(end,1) 0];
            
            subplot(1,2,i)
            stairs(both(:,1), both(:,2))
            
            % Plot the marginal CDFs
            g=both(1:end-1,2);
            x=both(1:end-1,1);
            
            bounds=xlim;
            x1=bounds(1);
            x2=bounds(2);
            
            g=[0;g;1];
            x=[x1;x;x2];
            
            
            cdfs(1,i)=g(1);
            for s=2:margCuts(i)+2
                delta=x(s)-x(s-1);
                cdfs(s, i)=cdfs(s-1, i)+ (delta*g(s-1));
            end
            
            hold on
            
            yyaxis right
            plot(x, cdfs(1:margCuts(i)+2, i))
            ylim([0 1.01])
            
            xlabel(['X', int2str(i)])
            
            legend('PDF', 'CDF')
        end
    end
    
    data=marginalCdfs;
    
    % Plot the 2D subregions
    figure
    plot(data(:,1), data(:,2), '.')
    if (copulaEnbl)
      title('Copula Data')
    else
      title('Original Data')  
    end
    xlabel('X1')
    ylabel('X2')
    xlim([min(data(:,1)), max(data(:,1))])
    ylim([min(data(:,2)), max(data(:,2))])
    
    leftBorder=min(data(:,1));
    rightBorder=max(data(:,1));
    
    bottomBorder=min(data(:,2));
    topBorder=max(data(:,2));
    
    figure
    axis([leftBorder rightBorder bottomBorder topBorder])
    plot(data(:,1), data(:,2),'.b')
    hold on
    t=copCuts;
    best_ptn=copulaCoords(:,:,1);
    
    for p=1:t
        x0=best_ptn(1,p);
        y0=best_ptn(3,p);
        w=best_ptn(2,p)-best_ptn(1,p);
        h=best_ptn(4,p)-best_ptn(3,p);
        dens(p)=subRegDensCopula(p);
    end
    
    for k=1:t
        x0=best_ptn(1,k);
        y0=best_ptn(3,k);
        w=best_ptn(2,k)-best_ptn(1,k);
        h=best_ptn(4,k)-best_ptn(3,k);
        R(k)=1- (dens(k)/max(dens));
        col=[1 R(k) 1];
        rectangle('Position', [x0 y0 w h]);
        hold on
    end
 
    title('BSP Cuts')
    xlabel('X1')
    ylabel('X2')
    xlim([min(data(:,1)), max(data(:,1))])
    ylim([min(data(:,2)), max(data(:,2))])
      
end

end

%% end plots >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc estimate_copula >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function f=estimate_copula(trees, testData, densMaps, VkMaps, coordsMarginal, cutsMarginal, coordsCopula, densMapCopula, cutsCopula)

data=testData;

N=size(testData, 1);
D=size(testData, 2);

g=zeros(N,D);
G=zeros(N,D);

estDensMap=zeros(N,1);
estDensFinal=zeros(N,1);

distCopula=zeros(N,1);

notInRange=0;

curTree={};

for dimension=1:D
    
    str1=['best_d', int2str(dimension)];
    load (trees, str1);   
    v = genvarname(str1);
    curTree=eval(v);
    curTree(end).label;
    densMap=densMaps(1:cutsMarginal(dimension), dimension);
    
    CDF=zeros(N,1);
    
    right=max(coordsMarginal(2,1:cutsMarginal(dimension),dimension));
    lastPtn= coordsMarginal(2,1:cutsMarginal(dimension),dimension)==right;
    
    
    ptnNumbers=findSubregion(curTree, data(:,dimension));
    
    for i=1:N
        
        z=data(i,dimension);
        
        ptnNumber=ptnNumbers(i);          

        if (ptnNumber~=-1)
            estDensMap(i)=densMap(ptnNumber);
        end
        
        
        %% Calc CDF for current dimension
        if(ptnNumber~=0)
            cdf=0;
            
            prev_ptns = find(coordsMarginal(2,1:cutsMarginal(dimension), dimension)<=z);
            
            for p=1:numel(prev_ptns)
                ptn=prev_ptns(p);
                delta=VkMaps(ptn, dimension) * densMap(ptn);
                cdf= cdf + delta;
                
            end
            
            ptn_begin=coordsMarginal(1,ptnNumber, dimension);
            delta2=(z - ptn_begin)*densMap(ptnNumber);
            cdf=cdf+delta2;
            
            CDF(i)=cdf;
            
        end
        
    end
    
    g(:, dimension)=estDensMap;
    G(:, dimension)=CDF;
    
end


%% Calc joint density of the Copula

data=G;

estDensMap=zeros(N,1);

notInRange2=0;

for i=1:N
    
    z=data(i,:);
    
    for j=1:cutsCopula
        bounds=reshape(coordsCopula(:,j),2, D);
        
        ptnNumber=0;
        cntr=0;
        for d=1:D
            if ( (z(d)<bounds(1,d)  ) || (z(d)>=bounds(2,d) ) )
                d=D+1;
                break
            else
                cntr=cntr+1;
            end
        end
        
        if(cntr==D)
            ptnNumber=j;
            j=cutsCopula+1;
            break
        end
    end
    
    if ((cntr==D) && (ptnNumber ~= 0))
        estDensMap(i)=densMapCopula(ptnNumber);
        
        distCopula(i)=ptnNumber;
        
    else
        notInRange2=notInRange2 + 1;
    end
    
end

for i=1:N
    estDensFinal(i) = estDensMap(i) * prod(g(i,:));
end

f = estDensFinal;

end

%% end estimate_copula >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc estimate_direct >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function f=estimate_direct(bestTrees, test_data, xt_MAP, dens_MAP, j_best)

data=test_data;

N=size(data, 1);
D=size(data, 2);
t=j_best;

est_dens_MAP=zeros(N,1);


for i=1:N
    
    z=data(i,:);
    
    for j=1:t
        bounds=reshape(xt_MAP(:,j),2,D);
        
        cntr=0;
        for d=1:D
            if ( (z(d)<bounds(1,d)  ) || (z(d)>bounds(2,d) ) )
                d=D+1;
                break
            else
                cntr=cntr+1;
            end
        end
        
        if(cntr==D)
            ptn_number=j;
            j=t+1;
            break
        end
    end
    
    if (cntr==D)
        est_dens_MAP(i)=dens_MAP(ptn_number);
    end
    
end

f = est_dens_MAP;

end

%% end estimate_direct >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc estimate_direct >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function f=estimate_direct1(test_data, xt_MAP, dens_MAP, j_best)

data=test_data;


N=size(data, 1);
D=size(data, 2);
t=j_best;

est_dens_MAP=zeros(N,1);

ptnNumbers=findSubregion(tree, data(:,dimension));

for i=1:N
    
    z=data(i,:);
    
    if (D==1)

        ptnNumber=ptnNumbers(i);

        if (ptnNumber~=-1)
            est_dens_MAP(i)=densMap(ptnNumber);
        end
        
    else
        
        for j=1:t
            bounds=reshape(xt_MAP(:,j),2,D);
            
            cntr=0;
            for d=1:D
                if ( (z(d)<bounds(1,d)  ) || (z(d)>bounds(2,d) ) )
                    d=D+1;
                    break
                else
                    cntr=cntr+1;
                end
            end
            
            if(cntr==D)
                ptn_number=j;
                j=t+1;
                break
            end
        end
        
        if (cntr==D)
            est_dens_MAP(i)=dens_MAP(ptn_number);
        end
        
    end
end

f = est_dens_MAP;

end


%% end estimate_direct >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc binTree >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function s=binTree(s, nextCut, n1, varargin)

d=size(s, 2);
left=s(nextCut).leftCoord;
n0=s(nextCut).sampleCount;
right=2*s(nextCut).mid - left;
val0=(left+right)/2;
val1={s(nextCut).label};
val2=n0;
val3=left;
val4={d+1};
val5={d+2};


s(nextCut)=struct('mid', val0, 'label', val1, 'sampleCount', val2, 'leftCoord', val3, 'leftChild', val4, 'rightChild', val5);


midPoint=s(nextCut).mid;
midPointL=0.5*(s(nextCut).mid + left);
midPointR=0.5*(s(nextCut).mid + right);
val0={midPointL, midPointR};
val1={s(nextCut).label, s(end).label+1};
val2={n1, n0-n1};
val3={left, midPoint};
val4={-1, -1};
val5={-1, -1};


s(end+1:end+2)=struct('mid', val0, 'label', val1, 'sampleCount', val2, 'leftCoord', val3, 'leftChild', val4, 'rightChild', val5);


end

%% end binTree >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


%% fnc findSubRegion >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function subregions=findSubregion(s, data)

N=length(data);

subregions=zeros(N,1);

for i=1:N
    
    x=data(i);
    
    p=1;            % Start with the root
    
    leaf=0;
    
    while (~leaf)   % Stop if it's a leaf node
        
        if (s(p).leftChild==-1)
            leaf=1;
        else
            midPoint=s(p).mid;
            
            if (x < midPoint)
                p=s(p).leftChild;
            else
                p=s(p).rightChild;
            end
        end
    end
    
    subregions(i)=s(p).label;
end

end


%% end findSubRegion >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


%% fnc parsave >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function parsave(fname, x)

save(fname, 'x')

end

%% end parsave >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc parload >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function tree=parload(treeName)

load(treeName);

end

%% end parload >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



%% fnc generate_data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function [data, actualDensities]=generate_data(distType,N,D,saveDens)

switch distType
    
    case 'G3'
        
    case 'mvn'
        
    case 'beta'
        
    case 'simpson'
        
end

s1='MVN_N';
if (D==1) s1='N'; end;
s2=int2str(N);
s3='_D';

s4=int2str(D);
s5='_densities';
fileName = [s1 s2 s3 s4];
fileName_densities = [s1 s2 s3 s4 s5];

data= zeros(N,D);  % stores sampled data point
actualDensities = zeros(N,1);

mu1_1 = [2 2];

Sigma1_1 = [0.04^2 0; 0 0.04^2];

mu1_2 = [3 3];

Sigma1_2 = [0.07^2 0; 0 0.07^2];

mu1_3 = [5 5];

Sigma1_3 = [0.04^2 0; 0 0.04^2];

mu3 = 0.5;
Sigma3 = 0.1;

mu4 = 0.4;
Sigma4 = 0.01;

mu5 = 0.6;
Sigma5 = 0.01;

% rng default  % For reproducibility

if (D==1)
    data=normrnd(mu3, Sigma3, N, 1);
    true_pdf = normpdf(data,mu3,Sigma3);
    
    save(fileName, 'data');
    actualDensities=true_pdf;
    save(fileName_densities, 'actualDensities');   
  
else
    
    for cnt=1:N
        r1=binornd(1, 0.75);
        if (r1==0)
            data(cnt,1:2) = mvnrnd(mu1_1,Sigma1_1)  ;
        else
            r2=binornd(1, (35/75));
            if (r2==0)
                data(cnt,1:2) = mvnrnd(mu1_2,Sigma1_2)  ;
            else
                data(cnt,1:2) = mvnrnd(mu1_3,Sigma1_3)  ;
            end
        end
    end
    
    
    if (D>=3)
        data(:,3)=normrnd(mu3, Sigma3, N, 1);
    end
    
    
    if (D>=4)
        for cnt=1:N
            for d=4:D
                bin=binornd(1, 0.6);
                if (bin==0)
                    data(cnt,d) = normrnd(mu4,Sigma4)  ;
                else
                    data(cnt,d) = normrnd(mu5,Sigma5)  ;
                end
            end
        end
    end
    
    save(fileName, 'data');
    
    if(saveDens==1)   % Calculate and save the PDF;
        
        for i=1:N
            true_pdf=ones(D,1);
            true_pdf(2) = 0.25*mvnpdf([data(i,1) data(i,2)],mu1_1,Sigma1_1) + 0.40*mvnpdf([data(i,1) data(i,2)],mu1_2,Sigma1_2) + 0.35*mvnpdf([data(i,1) data(i,2)],mu1_3,Sigma1_3);
            if (D>=3) true_pdf(3) = normpdf(data(i,3),mu3,Sigma3); end
            if (D>=4)
                for d=4:D
                    true_pdf(d) = 0.4 * normpdf(data(i,d), mu4, Sigma4) + 0.6 * normpdf(data(i,d),mu5, Sigma5);
                end
            end
            actualDensities(i,1) = prod(true_pdf);
        end
        
        save(fileName_densities, 'actualDensities');
    end
end
end

%% end generate_data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


function f=estimate_copula_old(testData, densMaps, VkMaps, coordsMarginal, cutsMarginal, coordsCopula, densMapCopula, cutsCopula)

data=testData;

N=size(testData, 1);
D=size(testData, 2);

g=zeros(N,D);
G=zeros(N,D);

estDensMap=zeros(N,1);
estDensFinal=zeros(N,1);

D1=1;

distCopula=zeros(N,1);

notInRange=0;

for dimension=1:D
    
    densMap=densMaps(1:cutsMarginal(dimension), dimension);
    
    CDF=zeros(N,1);
    
    right=max(coordsMarginal(2,1:cutsMarginal(dimension),dimension));
    lastPtn= coordsMarginal(2,1:cutsMarginal(dimension),dimension)==right;
    
    for i=1:N
        
        z=data(i,dimension);
        
        for j=1:cutsMarginal(dimension)
            bounds=coordsMarginal(:,j, dimension);
            
            ptnNumber=0;
            cntr=0;
            for d=1:D1
                if ( (z(d)<bounds(1,d)  ) || (z(d)>=bounds(2,d) ) )
                    d=D1+1;
                    break
                else
                    cntr=cntr+1;
                end
            end
            
            if(cntr==D1)
                ptnNumber=j;
                j=cutsMarginal(dimension)+1;
                break
            end
        end
        
        if ((cntr==D1) && (ptnNumber~=0))
            estDensMap(i)=densMap(ptnNumber);
            
        else
            if(z(1)==right)
                estDensMap(i)=densMap(lastPtn);
            end
            notInRange=notInRange + 1;
        end
        
        %% Calc CDF for current dimension
        if(ptnNumber~=0)
            cdf=0;
            
            prev_ptns = find(coordsMarginal(2,1:cutsMarginal(dimension), dimension)<=z);
            
            for p=1:numel(prev_ptns)
                ptn=prev_ptns(p);
                delta=VkMaps(ptn, dimension) * densMap(ptn);
                cdf= cdf + delta;
                
            end
            
            ptn_begin=coordsMarginal(1,ptnNumber, dimension);
            delta2=(z - ptn_begin)*densMap(ptnNumber);
            cdf=cdf+delta2;
            
            CDF(i)=cdf;
            
        end
        
    end
    
    g(:, dimension)=estDensMap;
    G(:, dimension)=CDF;
    
end


%% Calc joint density of the Copula

data=G;

notInRange2=0;

for i=1:N
    
    z=data(i,:);
    
    for j=1:cutsCopula
        bounds=reshape(coordsCopula(:,j),2, D);
        
        ptnNumber=0;
        cntr=0;
        for d=1:D
            if ( (z(d)<bounds(1,d)  ) || (z(d)>=bounds(2,d) ) )
                d=D+1;
                break
            else
                cntr=cntr+1;
            end
        end
        
        if(cntr==D)
            ptnNumber=j;
            j=cutsCopula+1;
            break
        end
    end
    
    if ((cntr==D) && (ptnNumber ~= 0))
        estDensMap(i)=densMapCopula(ptnNumber);
        
        distCopula(i)=ptnNumber;
        
    else
        notInRange2=notInRange2 + 1;
    end
    
end

for i=1:N
    estDensFinal(i) = estDensMap(i) * prod(g(i,:));
end

f = estDensFinal;

end




