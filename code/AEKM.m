function [ outG0, outFCell, outAlpha, outObj, outNumIter ] = AEKM( inXCell, inPara, inG0 ) 
% input: 
%       inXcell: v by 1 cell, and the size of each cell is d_v by n
%       inPara: parameter cell
%               inPara.maxIter: max number of iterator
%               inPara.thresh:  the convergence threshold
%               inPara.numCluster: the number cluster
%               inPara.r: the parameter to control the distribution of the
%                         weights for each view
%               inPara.reDim{v}: the reduced dimensionality
%       inG0: init common cluster indicator
% output:
%       outG0: the output cluster indicator (n by c)
%       outFcell: the cluster centroid for each view (d by c by v)
%       outObj: obj value
%       outNumIter: number of iterator
%       outAlpha: the weight of each view
% parameter settings
maxIter = inPara.maxIter;
thresh = inPara.thresh;
c = inPara.numCluster;
n = size(inXCell{1}, 2);
numView = length(inXCell);
% initilize alpha
alpha = ones(numView, 1)/numView; 
G0 = inG0;
for v = 1: numView    
    D4{v} = sparse(diag(ones(n, 1))* alpha(v));
end

%initialize W{v}
for v = 1: numView    
    W{v} = zeros(size(inXCell{v},1),inPara.reDim{v});
    idx = randperm(size(inXCell{v},1),size(inXCell{v},1));
    W{v}(sub2ind(size(W{v}),idx(1:inPara.reDim{v})',[1:inPara.reDim{v}]')) =1;
end

obj = zeros(maxIter, 1);

% loop
for t = 1: maxIter
    %step 1: update selection matrix
    GT_D_G = diag(1./diag(sparse(G0')*sparse(D4{v})*sparse(G0))+eps);
    
    %remove inf, which means some class is empty
    GT_D_G(find(GT_D_G == inf)) = 10^10;
    Sqrt_GT_D_G =GT_D_G^0.5;
    Sw{v} = inXCell{v}*sparse(D4{v})*sparse(G0)*Sqrt_GT_D_G;
    
    sqrt_D = spdiags(sqrt(sum(D4{v},2)),0,n,n);
    %diagonal points of (Sw), the quick version
    diag_pnts = sum((inXCell{v}*sparse(sqrt_D)).^2,2) - sum(Sw{v}.^2,2);
    [~,idx] = nth_element(diag_pnts,inPara.reDim{v}); % select nth small elements

    W{v} = zeros(size(inXCell{v},1),inPara.reDim{v});
    W{v}(sub2ind(size(W{v}),idx(1:inPara.reDim{v}),[1:inPara.reDim{v}]')) =1;
    
    %step 2: update class centroid matrix F{v}
    for v = 1: numView
        %M = (G0'*D4{v}*G0);
        M= diag(1./(diag(G0'*D4{v}*G0)+eps));%modified by wxd 
        N = sparse(W{v})'*inXCell{v}*D4{v}*G0;
        F{v} = N*M;
    end
   
    %step 3: update class indicator matrix G0  
    for i = 1:n
        dVec = zeros(numView, 1);
        for v = 1: numView
            xVec{v} = sparse(W{v})'*inXCell{v}(:,i);
            tt = diag(D4{v});
            dVec(v, 1) = tt(i);
        end
        G0(i,:) = searchBestIndicator(dVec, xVec, F);
    end
    
     % step 4: update alpha
    h = zeros(numView, 1);
    for v = 1: numView
        E{v} = (inXCell{v}'*sparse(W{v}) - sparse(G0)*F{v}');
        Ei2{v} = sqrt ( sum( sqrt(sum(E{v}.*E{v}, 2) +eps)) +eps);
        alpha(v) = 0.5./Ei2{v};
    end
    
     % step 5: update D4
    h = zeros(numView, 1);
    for v = 1: numView
        E{v} =(inXCell{v}'*sparse(W{v}) - sparse(G0)*F{v}');
        Ei2{v} = sqrt(sum(E{v}.*E{v}, 2) + eps);                
        D4{v} = sparse(diag(0.5./Ei2{v}*(alpha(v))));
    end
    
    %step 6: calculate the obj
     % calculate the obj
    obj(t) = 0;
    for v = 1: numView
        obj(t) = obj(t) + sqrt(sum(Ei2{v}));
    end
    if(t > 1)
        diff = obj(t-1) - obj(t);
        if(diff < thresh)
            break;
        end
    end
end


outObj = obj;
outNumIter = t;
outFCell = F;
outG0 = G0;
outAlpha = alpha;
%plot(obj)
%obj

end
%% function searchBestIndicator
function outVec = searchBestIndicator(dVec, xCell, F)
% solve the following problem,
numView = length(F);
c = size(F{1}, 2);
tmp = eye(c);
obj = zeros(c, 1);
for j = 1: c
    for v = 1: numView
        obj(j,1) = obj(j,1) + dVec(v) * (norm(xCell{v} - F{v}(:,j))^2);
    end
end
[min_val, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end


