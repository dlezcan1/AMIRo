function [pos, wv, Rmat] = singlebend_singlelayer_shape(...
    kc, w_init, L, needle_mech_params, theta0, ds ...
)
    arguments
        kc double
        w_init (3, 1)
        L double
        needle_mech_params struct
        theta0 double = 0;
        ds double = 0.5;
    end
    
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    % needle shape
    s = 0:ds:L;
    k0 = kc * (1 - s/L).^2;
    k0_prime = -2 * kc/L * (1 - s/L);
    
    w0       = [k0;       zeros(2, length(k0))];
    w0_prime = [k0_prime; zeros(2, length(k0_prime))];
    
    [wv, pos, Rmat] = fn_intgEP_w0_Dimitri(...
        w_init, w0, w0_prime, theta0, 0, ds, length(s), B, Binv...
    );
    
    
end