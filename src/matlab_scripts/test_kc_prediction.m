function [p, a, c, w1, w2, ab] = test_kc_prediction(L_data, kc_data, kc_errors, L_test)
    
    %% Determine kc_init and L_init
    [L_init, min_idx] = min(L_data);
    kc_init = kc_data(min_idx);
    
    
    %% Prepare ratios
    % data
    kc_data_ratios = kc_data./kc_init;
    L_data_ratios = L_data./L_init;
    
    % test
    L_test_ratios = L_test./L_init;
    
    %% Oringinal prediction
    
    kc_test_og_p = kc_init .*(L_test_ratios).^(-0.592);
    kc_test_linear = kc_init .* (L_test_ratios);

    %% Test (L/L_ref)^p | log-log lst squares fit
    disp("(L_ref/L)^p");
    
    p = log(L_data_ratios) \ log(kc_data_ratios);
    
    kc_test_p = kc_init .* (L_test_ratios).^p;
    
    fprintf("p = %f\n\n", p);
    
    %% Test a(L/L_ref)^2 + (1-a) (L/L_ref)
    disp("a(L/L_ref)^2 + (1-a) (L/L_ref)");
    X = L_data_ratios.^2 - L_data_ratios;
    Y = kc_data_ratios - L_data_ratios;
    
    a = X\Y;
    
    kc_test_a = kc_init*(a * L_test_ratios.^2 + (1-a).*(L_test_ratios));
    
    fprintf("a = %f\n\n", a);
    
    %% Test 1 + c*ln(L/L_ref)
    disp("1 + c*ln(L/L_ref)");
    X = log(L_data_ratios);
    Y = kc_data_ratios - 1;
    c = X\Y;
    
    kc_test_c = kc_init*(1 + c*log(L_test_ratios));
    
    fprintf("c = %d\n\n", c);
    
    %% Test 2/(1 + exp(-w (L - L_ref)))
    disp("2/(1 + exp(-w (L - L_ref)))")
    
    X = L_init - L_data;
    Y = log(2./kc_data_ratios - 1);
    w1 = X\Y;
    
    kc_test_w1 = kc_init*2./( 1 + exp(-w1 * (L_test - L_init)));
    
    fprintf('w = %f\n\n', w1);
    
    %% Test 2/(1 + exp(-w (L - L_ref)/L))
    disp("2/(1 + exp(-w (L - L_ref)/L))")
    
    X = 1./L_data_ratios - 1;
    Y = log(2./kc_data_ratios - 1);
    w2 = X\Y;
    
    kc_test_w2 = kc_init*2./( 1 + exp(-w2 * (L_test - L_init)./L_test));
    
    fprintf('w = %f\n\n', w2);
    
    %% Test a(dL/L_ref)^2 + b(dL/L_ref) + 1
    disp("a(dL/L_ref)^2 + b(dL/L_ref) + 1");
    
    X = [(L_data_ratios - 1).^2, (L_data_ratios - 1)];
    Y = kc_data_ratios - 1;
    
    ab = X\Y;
    a = ab(1); b = ab(2);
    
    fprintf("a = %f\nb = %f\n\n", a,b);
    kc_test_ab = kc_init*(a*(L_test_ratios - 1).^2 + b*(L_test_ratios - 1) + 1);
    
    %% Plotting
    errorbar(L_data, kc_data, kc_errors, '*', 'DisplayName', 'Data'); hold on;
    plot(L_test, kc_test_linear, 'DisplayName', '\kappa_{c,ref}(L/L_{ref})'); hold on;
    plot(L_test, kc_test_og_p, 'DisplayName', '\kappa_{c,ref}(L_{ref}/L)^{0.592}'); hold on;
    plot(L_test, kc_test_p, 'DisplayName', '\kappa_{c,ref}(L/L_{ref})^p'); hold on;
    plot(L_test, kc_test_a, 'DisplayName', '\kappa_{c,ref}(a(L/L_{ref})^2 + (1-a)(L/L_{ref}))'); hold on;
    plot(L_test, kc_test_c, 'DisplayName', '\kappa_{c,ref}(1 + c log(L/L_{ref}))'); hold on;
    plot(L_test, kc_test_w1, 'DisplayName', '\kappa_{c,ref}(2/(1 + exp(-w(L-L_{ref})))'); hold on;
    plot(L_test, kc_test_w2, 'DisplayName', '\kappa_{c,ref}(2/(1 + exp(-w(L-L_{ref})/L))'); hold on;
    plot(L_test, kc_test_ab, 'DisplayName', '\kappa_{c,ref}(a(L-L_{ref}/L_{ref})^2 + b(L-L_{ref}/L_{ref}) + 1)'); hold on;
    xlim([min(L_data) - 5, max(L_data) + 5]);
    hold off;
    xlabel('Insertion Depth (mm)'); ylabel('\kappa_c (1/mm)');
    legend('location', 'bestoutside');
    
end
    
    
    
    
    
    