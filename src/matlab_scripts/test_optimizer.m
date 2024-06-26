clear;
kc_act = 0.001;
w_init_act = [0.005, 0.003, -0.01]';

<<<<<<< Updated upstream
L = 35;
s_meas = [10, 45, 80, 100];
=======
L = 200;
s_meas = [10, 30, 65, 100];
>>>>>>> Stashed changes
ds = 0.5;
s = 0:ds:L;
s_idx_aa = find(any(s' == s_meas, 2));
N = numel(s);

k0 = kc_act * (1 - s/L).^2;
k0prime = -2*kc_act/L*(1 - s/L);

w0 = [k0; zeros(2,N)];
w0prime = [k0prime; zeros(2,N)];

B = diag([25539.64040739, 25539.64040739, 19798.17085844]);
Binv = inv(B);

[wv, pmat, ~] = fn_intgEP_w0_Dimitri(w_init_act, w0, w0prime, 0, 0, ds, N, B, Binv);

curvs_aa = wv(1:2, s_idx_aa);
weights = ones(1, length(s_idx_aa));

<<<<<<< Updated upstream
%% 
curvs_aa = [
    8.927878507420123e-05, ...
    -0.0005695292362794418; ...
    -0.00023095939876480897, ...
    -0.00017777582321463223; ...
    -0.0007775240953053716, ...
    0.0027939044884432622; ...
    0.0009992215395756168, ...
    0.007964753048007707
]'
L = 35;
s_aa = [10, 30, 65, 100];
mask_in = s_aa <= L;

=======
curvs_aa = [
    -0.00010360637616939758, -0.0009423841218937366;
    2.790138306940568e-05, -0.00028069456751391017;
    4.1899078345001015e-06, 4.784952322792695e-07;
    -1.2163316166428914e-05, -3.564428214474146e-05
]';

curvs_aa = [curvs_aa(2, :); curvs_aa(1, :)]
>>>>>>> Stashed changes



%% Optimizer
w_init_i = zeros(3,1);
kc_i = kc_act/2;
eta = [w_init_i; kc_i];
scalef0 = 1;
Cval = costfn_shape_singlebend(eta, curvs_aa, s_idx_aa, ds, N, B, Binv, scalef0, weights);
scalef = 1/Cval;

% optimization
x0 = eta; % initial value
LB = [-0.01*ones(2,1); -0.001; 0]; % lower bound
UB = [0.01*ones(2,1); -LB(3); 0.01]; % upper bound

oldopts = optimset('fmincon');
options = optimset(oldopts,'Algorithm','interior-point','TolFun',1e-8,'TolX',1e-8,...
    'MaxFunEvals',10000, 'Display', 'on');
[x, fval, exitflag] = fmincon( @(x) costfn_shape_singlebend(x, curvs_aa,...
    s_idx_aa, ds, length(s), B, Binv, scalef, weights),...
    x0, [], [], [], [], LB, UB, [], options);

w_init = x(1:3);
kc = x(4);
disp("w_init = ")
disp(w_init');
fprintf("kc = %.9f\n", kc);
% fprintf("error kc = %.9f\n", kc - kc_act);
% disp("Error in w_init")
% disp((w_init - w_init_act)');

%% Helper functions
function y = costfn_shape_singlebend(eta,data,s_index_meas,ds,N,B,Binv,scalef,weights) 
    weights = weights(1:numel(s_index_meas));
    weights = weights/sum(weights, 'all');
    % unpack the variables
    w_init = eta(1:3); 
    kc = eta(4); 

    % arclength parameters
    L = (N-1)*ds; % in mm 
    s = [0:ds:L]; 

    % intrinsic curvature (quadratic) 
    k0 = kc*(1 - s/L).^2; 
    w0 = [k0;zeros(1,N);zeros(1,N)]; 

    k0prime = -2*kc/L*(1 - s/L); 
    w0prime = [k0prime;zeros(1,N);zeros(1,N)]; 

    % integration of the E-P equation 
    wv = fn_intgEP_w0_Dimitri(w_init, w0, w0prime,0,0,ds,N,B,Binv);

    % exclude torsion 
    yv = wv(1:2,s_index_meas) - data(1:2,:); 
    y = norm(yv.*weights,'fro')^2*scalef; 

end