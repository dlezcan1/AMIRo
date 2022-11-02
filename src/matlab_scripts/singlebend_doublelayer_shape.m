function [pos, wv, Rmat] = singlebend_doublelayer_shape(...
    kc1, kc2, w_init, L, z_crit, needle_mech_params, theta0, ds ...
)
    arguments
        kc1 double
        kc2 double
        w_init (3,1)
        L double
        z_crit double
        needle_mech_params struct
        theta0 double = 0;
        ds double = 0.5;
    end
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    s = 0:ds:L;
    
    [wv, pos, Rmat, s_crit] = fn_intgEP_zcrit_2layers_Dimitri(...
        w_init, kc1, kc2, z_crit, ...
        theta0, 0, ds, length(s), B, Binv ...
    );
    

end