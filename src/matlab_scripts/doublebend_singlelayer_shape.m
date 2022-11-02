function [pos, wv, Rmat] = doublebend_singlelayer_shape(...
    kc, w_init, L, s_dbl_bend, needle_mech_params, theta0, ds ...
)
    arguments
        kc double
        w_init (3,1)
        L double
        s_dbl_bend double
        needle_mech_params struct
        theta0 double = 0
        ds double = 0.5
    end
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    if L <= s_dbl_bend
        [pos, wv, Rmat] = singlebend_singlelayer_shape(...
            kc, w_init, L, needle_mech_params, ...
            theta0, ds ...
        );
        
        if L == s_dbl_bend
            theta_z = pi;
        else
            theta_z = 0;
        end
    else
        s = 0:ds:L;
        [~, s_idx_turn] = min(abs(s - s_dbl_bend));
        s1 = s(1:s_idx_turn);
        s2 = s(s_idx_turn:end);

        kc1 = kc*((s1(end) - s1(1))/L)^(2/3);
        kc2 = kc*((s2(end) - s2(1))/L)^(2/3);

        % intrinsic curvature kappa_0 (quadratic)
        k0_1    = kc1*(1 - s1/L).^2;
        k0_2    = -kc2*(1 - s2/L).^2;
        k0_turn = (k0_1(end) + k0_2(1))/2; %0;
        k0      = [k0_1(1:end-1),k0_turn,k0_2(2:end)];

        k0_prime1     = -2*kc1/L*(1 - s1/L);
        k0_prime2     = 2*kc2/L*(1 - s2/L);
        k0_prime_peak = (k0_2(2) - k0_1(end-1))/2/ds; 
        k0_prime      = [k0_prime1(1:end-1),k0_prime_peak,k0_prime2(2:end)];
        
        % intrinsic curvature \omega_0
        w0       = [k0;       zeros(2, length(s))];
        w0_prime = [k0_prime; zeros(2, length(s))];
        
        [wv, pos, Rmat] = fn_intgEP_w0_Dimitri( ...
            w_init, w0, w0_prime, theta0, 0, ds, length(s), B, Binv ...
        );
    
        theta_z = pi;
        
    end
    
    % rotate by pi
    Rz = Rot_z(theta_z);
    pos = Rz * pos;
    Rmat = pagemtimes(Rz, Rmat);
    
end