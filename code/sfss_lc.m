% written by xiaodong wang
function W = sfss_lc(X,Y,para,A0,U)
%%
alpha = para.alpha;
beta = para.beta;
gamma = para.gamma;
rd = para.rd;

[dim,num] = size(X);
class = size(Y,2);
Id = eye(dim);
In = eye(num);

W = rand(dim,class);
F = rand(num,class);
iter = 1;
obji = 1;
show_matrix_as_image(F);
dis_F = sqrt(abs(L2_distance_1(F',F'))+eps);
while 1
    A = A0./(2*dis_F);
    A=(A+A')/2;
    L=diag(sum(A,2))-A;
    
    d = 0.5./sqrt(sum(W.*W,2)+eps);
    Dw = diag(d);
    B=L+U+gamma*In;
    M = X*X'+alpha*Dw+beta*Id;
    N=M-gamma*X*inv(B)*X';
    C = Id-beta*inv(N)+eps;
    D= inv(N)*X*inv(B)*U*Y;
    D=D*D';
    [eigvec eigval] = eig(C\D);
    [eigval,idx] = sort(diag(eigval),'descend');
    Q = eigvec(:,idx(1:class-rd));
    %Q = real(Q);
    Q=orth(Q);
    
    F=(B-gamma*X'*inv(M-beta*Q*Q')*X+eps)\(U*Y);
    show_matrix_as_image(F);
    W = (M-beta*Q*Q'+eps)\(X*F);
    dis_F = sqrt(abs(L2_distance_1(F',F'))+eps);
    %objective(iter) = sum(sqrt(sum((X'*W-Y).*(X'*W-Y),2)+eps))+alpha*sum(sqrt(sum(W.*W,2)+eps))+beta*(norm((W-Q*Q'*W),'fro'))^2;
    objective(iter) = 0.5*sum(sum((A0.*dis_F)))+trace((F-Y)'*U*(F-Y))+gamma*(alpha*sum(sqrt(sum(W.*W,2)))+beta*(norm((W-Q*Q'*W),'fro'))^2+(norm((X'*W-F),'fro'))^2);
    cver = abs((objective(iter)-obji)/obji);
    obji = objective(iter);
    iter = iter+1;
    if (cver < 10^-3 && iter > 2) || iter ==30, break, end
end



plot(objective,'-g','LineWidth',1.2);
set(gca,'XTick',0:2:100); 
%set(gca,'YTick',0:2:400); 
set(gca,'FontSize',14);
xlabel('number of iterations ');
ylabel('objective function value') ;
