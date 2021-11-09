%% GalilController
%
% wrapper class for the galil controller used for 4 stage (XYZ,LS) needle
% insertion robot
%
% - written by: Dimitri Lezcano

classdef GalilController
    properties (Access=private)
        X_count_per_mm  = 2000;
        Y_count_per_mm  = 2000;
        Z_count_per_mm  = 2000;
        LS_count_per_mm = 43680;
        axes_labels = ["B", "C", "D", "E"]; % X, Y, Z, LS
    end 
    properties (SetAccess=protected)
        ip_address      = '192.168.1.201';
        controller; % galil controller
    end 
    properties (Access=public)
        waitCompletion logical = false;
    end
    methods (Access=protected)
        function galil_prompt(obj, msg)
           disp(strcat("Galil: ", msg));
        end
        function msg = format_msg_axes(obj, X, Y, Z, LS)
           arguments
               obj GalilController;
               X = '';
               Y = '';
               Z = '';
               LS = '';
           end
           
           msg = "";
            
        end
    end
    methods 
        % constructor
        function obj = GalilController(ip_address)
            if nargin == 1
                obj.ip_address = ip_address;
            end
            
            obj.controller = actxserver('galil');%set the variable obj.controller to the GalilTools COM wrapper
            response = obj.controller.libraryVersion;%Retrieve the GalilTools library versions
            disp(response);%display GalilTools library version
            obj.controller.address = obj.ip_address;%Open connections dialog box
            response = obj.controller.command(strcat(char(18), char(22)));%Send ^R^V to query controller model number
            disp(strcat('Connected to: ', response));%print response
            response = obj.controller.command('MG_BN');%Send MG_BN command to query controller for serial number
            disp(strcat('Serial Number: ', response));%print response            
            
            % set parameter of controller
            % axis B: X direction of XYZ robot    2000 unit: 1 mm
            % axis C: Y direction of XYZ robot    2000 unit: 1 mm
            % axis D: Z direction of XYZ robot    2000 unit: 1 mm
            % axis E: linear stage                43680 unit: 1 mm
            obj.controller.command('KP,54,15,54,25');
            obj.controller.command('KI,4,4,4,4');
            obj.controller.command('KD,480,332,480,480');
            obj.controller.command('AC,3000,3000,3000,10000');
            obj.controller.command('DC,3000,3000,3000,10000');
%             obj.controller.command('SP,43680,5000,5000,5000');
            obj.controller.command('SP,5000,5000,5000,87360');
            
            % Set timer to 1000 ms 
            obj.controller.command("TWA=1000");
        end
        
        function abort(obj)
            obj.galil_prompt("AB | ABORT!");
            obj.controller.command("AB");
        end
        
        function allMotorsOff(obj)
            obj.galil_prompt("MO");
            obj.controller.command("MO");
        end
        
        function allMotorsOn(obj)
            obj.galil_prompt("SH");
            obj.controller.command("SH");
        end
        
        % get the relative encoder position in mm
        function pos = currentPosition(obj)
            pos_msg = obj.controller.command("PA ,?,?,?,?");
%             obj.galil_prompt("PA ,?,?,?,?");
            pos_counts = str2num(pos_msg);
            
            pos = pos_counts ./ [obj.X_count_per_mm, obj.Y_count_per_mm, obj.Z_count_per_mm, obj.LS_count_per_mm];
            
        end
        
        % move absolute position in mm
        function moveAbsolute(obj, x, y, z, ls)
           arguments
              obj;
              x {mustBeNumeric} = 0;
              y {mustBeNumeric} = 0;
              z {mustBeNumeric} = 0;
              ls {mustBeNumeric} = 0;
           end
           % Movement order: N/A, X, Y, Z, LS
           base_move_cmd = "PA ,%d,%d,%d,%d"; % relative movements
           
           % perform the calculations for counts
           x_counts = round(obj.X_count_per_mm * x);
           y_counts = round(obj.Y_count_per_mm * y);
           z_counts = round(obj.Z_count_per_mm * z);
           ls_counts = round(obj.LS_count_per_mm * ls);

           % generate the move command
           move_cmd = sprintf(base_move_cmd, x_counts, y_counts, ...
                                             z_counts, ls_counts);
           move_cmd = strrep(move_cmd, ',0', ','); % remove 0's
           
           % the Begin and move complete command
           begin_cmd = "BG ";
           mvcmp_cmd = "MC ";
           
           if ls_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(4));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(4));
           end
           if y_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(2));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(2));
           end
           if z_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(3));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(3));
           end
           if x_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(1));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(1));
           end
           
           % move the robot and wait for completion
           obj.controller.command(move_cmd);
           obj.galil_prompt(move_cmd);
           obj.controller.command(begin_cmd);
           obj.galil_prompt(begin_cmd);
           if obj.waitCompletion
               obj.controller.command(mvcmp_cmd);
               obj.galil_prompt(mvcmp_cmd);
           end
           
           % wait until movement is finished
           while false && obj.waitCompletion
               try
                   obj.controller.command("MG 1");
                   break;
               catch
                   continue;
               end
           end
        end
                       
        % move relative position in mm
        function moveRelative(obj, x, y, z, ls)
           arguments
              obj;
              x {mustBeNumeric} = 0;
              y {mustBeNumeric} = 0;
              z {mustBeNumeric} = 0;
              ls {mustBeNumeric} = 0;
           end
           % Movement order: N/A, X, Y, Z, LS
           base_move_cmd = "PR ,%d,%d,%d,%d"; % relative movements
           
           % perform the calculations for counts
           x_counts = round(obj.X_count_per_mm * x);
           y_counts = round(obj.Y_count_per_mm * y);
           z_counts = round(obj.Z_count_per_mm * z);
           ls_counts = round(obj.LS_count_per_mm * ls);

           % generate the move command
           move_cmd = sprintf(base_move_cmd, x_counts, y_counts, ...
                                             z_counts, ls_counts);
           move_cmd = strrep(move_cmd, ',0', ','); % remove 0's
           
           % the Begin and move complete command
           begin_cmd = "BG ";
           mvcmp_cmd = "MC ";
           
           if ls_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(4));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(4));
           end
           if y_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(2));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(2));
           end
           if z_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(3));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(3));
           end
           if x_counts ~= 0
               begin_cmd = strcat(begin_cmd, obj.axes_labels(1));
               mvcmp_cmd = strcat(mvcmp_cmd, obj.axes_labels(1));
           end
           
           % move the robot and wait for completion
           obj.controller.command(move_cmd);
           obj.galil_prompt(move_cmd);
           obj.controller.command(begin_cmd);
           obj.galil_prompt(begin_cmd);
           if obj.waitCompletion
               obj.controller.command(mvcmp_cmd);
               obj.galil_prompt(mvcmp_cmd);
           end
           
           % wait until movement is finished
           while false && obj.waitCompletion
               try
                   obj.controller.command("MG 1");
                   break;
               catch
                   continue;
               end
           end
        end
        function motorsOff(obj, X, Y, Z, LS)
            arguments
                obj GalilController;
                X logical = false;
                Y logical = false;
                Z logical = false;
                LS logical = false;
            end
            if ~any([X,Y,Z,LS])
                return;
            else
                msg = strcat("MO ", obj.axes_labels{[X, Y, Z, LS]});
                obj.galil_prompt(msg);
                obj.controller.command(msg);
            end
            
        end
        function motorsOn(obj, X, Y, Z, LS)
            arguments
                obj GalilController;
                X logical = false;
                Y logical = false;
                Z logical = false;
                LS logical = false;
            end
            
            if ~any([X, Y, Z, LS])
                return;
            else
                msg = strcat("SH ", obj.axes_labels{[X, Y, Z, LS]});
                obj.galil_prompt(msg);
                obj.controller.command(msg);
            end
        end
        
        function stopAxes(obj, X, Y, Z, LS)
            arguments
                obj GalilController;
                X logical = false;
                Y logical = false;
                Z logical = false;
                LS logical = false;
            end
            if ~any([X, Y, Z, LS])
                return;
            else
                msg = strcat("ST ", obj.axes_labels{[X, Y, Z, LS]});
                obj.galil_prompt(msg);
                obj.controller.command(msg);
            end
        end
        
        % zero the axes
        function zeroAllAxes(obj)
            obj.zeroAxes(true, true, true, true);
        end
        
        % zero the axes
        function zeroAxes(obj, X, Y, Z, LS)
            arguments
                obj GalilController;
                X logical = false;
                Y logical = false;
                Z logical = false;
                LS logical = false;
            end
            
            if ~any([X, Y, Z, LS])
                return;
            end
            
            msg = "DP ,%s,%s,%s,%s";
            if X
                msg_x = '0';
            else
                msg_x = '';
            end
            if Y
                msg_y = '0';
            else
                msg_y = '';
            end
            if Z
                msg_z = '0';
            else
                msg_z = '';
            end
            if LS
                msg_ls = '0';
            else
                msg_ls = '';
            end
            
            msg = sprintf(msg, msg_x, msg_y, msg_z, msg_ls);
            
            obj.galil_prompt(msg);
            obj.controller.command("DP ,0,0,0,0");
        end
    end
end