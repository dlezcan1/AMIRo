classdef MCFNeedle < FBGNeedle
    %MCFNeedle: FBGNeedle wrapper for MCF needles
    %   Detailed explanation goes here

    properties
        central_core_ch int32 % the central core channel
    end

    methods
        function obj = MCFNeedle(json_file)
            obj@FBGNeedle(json_file);
            obj.central_core_ch = obj.json_obj.CentralCoreCH;
        end

        function wl_shifts_tcomp = temperature_compensate(obj, wl_shifts)
            arguments
                obj MCFNeedle
                wl_shifts (:, :)
            end
            
            wl_shifts_tcomp = zeros(size(wl_shifts));
            
            for aa_i = 1:obj.num_active_areas
                aa_mask = obj.active_area_mask(aa_i) & ~obj.central_core_mask();
                
                wl_shifts_aa_mean = mean(wl_shifts{:, aa_mask}, 2);

                wl_shifts_tcomp(:, aa_mask) = ( ...
                    wl_shifts{:, aa_mask}...
                    - wl_shifts_aa_mean ...
                );
            end
        end 

        function cc_mask = central_core_mask(obj)
            cc_mask = obj.channel_mask(obj.central_core_ch);

        end

    end % methods (instance)
end % class: MCFNeedle