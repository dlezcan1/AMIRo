%% FBGNeedle.m
% a class for FBGNeedle parametrizations
%
% not needed, use the py class to import from needle_shape_sensing python
% package!
%
% - written by: Dimitri Lezcano

classdef FBGNeedle
    properties
        serial_number string        % serial number of needle
        length double               % length of needle
        num_channels int32          % number of channels
        num_active_areas int32      % number of AA's
        sensor_locations struct     % the sensor locations measured from the base
        calibration_matrices struct % associated calibration mats
        weights struct
        json_obj struct             % the json struct from parsing the file
    end
    methods
        function obj = FBGNeedle(json_file)
            % load json information
            fid = fopen(json_file);
            assert(fid > 0, sprintf("File %s: does not exist", json_file));
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);
            obj.json_obj = jsondecode(str);

            % parse json information
            obj.serial_number    = obj.json_obj.serialNumber;
            obj.num_active_areas = obj.json_obj.x_ActiveAreas;
            obj.num_channels     = obj.json_obj.x_Channels;
            obj.length           = obj.json_obj.length;

            obj.calibration_matrices = obj.json_obj.CalibrationMatrices;
            obj.sensor_locations     = obj.json_obj.SensorLocations;

            if isfield(obj.json_obj, 'weights')
                obj.weights = obj.json_obj.weights;
            end
        end 

        function slocs = array_sensor_locations(obj)
            slocs = cellfun(...
                @(x)(obj.json_obj.SensorLocations.(x)), ...
                fields(obj.json_obj.SensorLocations) ...
            );

        end

        function wl_shifts_tcomp = temperature_compensate(obj, wl_shifts)
            arguments
                obj FBGNeedle
                wl_shifts (:, :) double
            end
            
            wl_shifts_tcomp = zeros(size(wl_shifts));
            
            for aa_i = 1:obj.num_active_areas
                aa_mask = obj.active_area_mask(aa_i);
                
                wl_shifts_aa_mean = mean(wl_shifts(:, aa_mask), 2);

                wl_shifts_tcomp(:, aa_mask) = ( ...
                    wl_shifts(:, aa_mask)...
                    - wl_shifts_aa_mean ...
                );
            end
        end

        function ch_aa = generate_chaa(obj)
            ch_aa = FBGNeedle.generate_ch_aa(obj.num_channels, obj.num_active_areas);
        end

        function num_sensors = num_sensors(obj)
            num_sensors = obj.num_channels * obj.num_active_areas;
        end
        
        function ch_mask = channel_mask(obj, ch)
            ch_mask = FBGNeedle.ch_mask(ch, obj.num_channels, obj.num_active_areas);
        end

        function aa_mask = active_area_mask(obj, aa)
            aa_mask = FBGNeedle.aa_mask(aa, obj.num_channels, obj.num_active_areas);
        end

        function chaa_mask = sensor_mask(obj, ch, aa)
            chaa_mask = obj.channel_mask(ch) & obj.active_area_mask(aa);
        end
    
    end % methods (instance)

    methods(Static, Access = protected)
        function ch_aa = generate_ch_aa(num_channels, num_aas)
            chs = split(sprintf("CH%d\n", 1:num_channels));
            chs = chs(1:num_channels);

            aas = split(sprintf("AA%d\n", 1:num_aas));
            aas = aas(1:num_aas);

            ch_aa = reshape(chs' + " | "  + aas, 1, []);

        end

        function ch_mask = ch_mask(ch, num_channels, num_aas)
            ch_aa = FBGNeedle.generate_ch_aa(num_channels, num_aas);

            ch_mask = contains(ch_aa, sprintf("CH%d", ch));
        end

        function aa_mask = aa_mask(aa, num_channels, num_aas)
            ch_aa = FBGNeedle.generate_ch_aa(num_channels, num_aas);

            aa_mask = contains(ch_aa, sprintf("AA%d", aa));
        end


    end % methods (static, protected)

    methods(Static, Access = public)
        function obj = load_json(json_file)
            try
                obj = MCFNeedle(json_file);
                return
            catch
                ...
            end

            obj = FBGNeedle(json_file);

        end
    end % methods (static, public)
end % class: FBGNeedle