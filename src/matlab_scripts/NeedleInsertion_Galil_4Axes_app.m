classdef NeedleInsertion_Galil_4Axes_app < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        TabGroup                        matlab.ui.container.TabGroup
        RobotControlTab                 matlab.ui.container.Tab
        GridLayout                      matlab.ui.container.GridLayout
        ABORTButton                     matlab.ui.control.Button
        MovementDirectionLabel          matlab.ui.control.Label
        LSAxisDirection                 matlab.ui.container.ButtonGroup
        LSRetractButton                 matlab.ui.control.RadioButton
        LSInsertButton                  matlab.ui.control.RadioButton
        ZAxisDirection                  matlab.ui.container.ButtonGroup
        ZDowntButton                    matlab.ui.control.RadioButton
        ZUpButton                       matlab.ui.control.RadioButton
        YAxisDirection                  matlab.ui.container.ButtonGroup
        YAwayButton                     matlab.ui.control.RadioButton
        YTowardsButton                  matlab.ui.control.RadioButton
        XAxisDirection                  matlab.ui.container.ButtonGroup
        XRetractButton                  matlab.ui.control.RadioButton
        XInsertButton                   matlab.ui.control.RadioButton
        ZeroAxesButton                  matlab.ui.control.Button
        MovementLabel                   matlab.ui.control.Label
        LinearStageMoveEditField        matlab.ui.control.NumericEditField
        ZAxisMoveEditField              matlab.ui.control.NumericEditField
        YAxisMoveEditField              matlab.ui.control.NumericEditField
        XAxisMoveEditField              matlab.ui.control.NumericEditField
        MovementTypeButtonGroup         matlab.ui.container.ButtonGroup
        SaveRobotConfigButton           matlab.ui.control.Button
        AbsoluteButton                  matlab.ui.control.RadioButton
        RelativeButton                  matlab.ui.control.RadioButton
        CurrentPositionLabel            matlab.ui.control.Label
        RobotAxisLabel                  matlab.ui.control.Label
        MoveLSButton                    matlab.ui.control.Button
        MoveZButton                     matlab.ui.control.Button
        MoveAllButton                   matlab.ui.control.Button
        MoveYButton                     matlab.ui.control.Button
        MoveXButton                     matlab.ui.control.Button
        LinearStageEditField            matlab.ui.control.NumericEditField
        LinearStageinsertionEditFieldLabel  matlab.ui.control.Label
        ZAxisEditField                  matlab.ui.control.NumericEditField
        ZAxisdownEditFieldLabel         matlab.ui.control.Label
        YAxisEditField                  matlab.ui.control.NumericEditField
        YAxistowardsyouLabel            matlab.ui.control.Label
        XAxisEditField                  matlab.ui.control.NumericEditField
        XAxisinsertionEditFieldLabel    matlab.ui.control.Label
        StereoImagesTab                 matlab.ui.container.Tab
        StereoImageAxes                 matlab.ui.control.UIAxes
        InterrogatorConnectionPanel     matlab.ui.container.Panel
        InterrogatorIPEditField         matlab.ui.control.EditField
        InterrgatorIPLabel              matlab.ui.control.Label
        InterrogatorConnectLamp         matlab.ui.control.Lamp
        InterrogatorDisconnectButton    matlab.ui.control.Button
        InterrogatorConnectButton       matlab.ui.control.Button
        DataCollectionFBGandStereoCameraPanel  matlab.ui.container.Panel
        GridLayout2                     matlab.ui.container.GridLayout
        ClearStatusButton               matlab.ui.control.Button
        DataStatusTextArea              matlab.ui.control.TextArea
        StatusTextAreaLabel             matlab.ui.control.Label
        SensorIDRunEditFieldLabel       matlab.ui.control.Label
        SensorIDRunEditField            matlab.ui.control.EditField
        InsertionDepthmmEditFieldLabel  matlab.ui.control.Label
        InsertionDepthEditField         matlab.ui.control.NumericEditField
        InsertionHoleSpinnerLabel       matlab.ui.control.Label
        InsertionHoleSpinner            matlab.ui.control.Spinner
        SaveDirectoryEditFieldLabel     matlab.ui.control.Label
        SaveDirectoryEditField          matlab.ui.control.EditField
        FBGSamplesEditFieldLabel        matlab.ui.control.Label
        NumberFBGSamplesEditField       matlab.ui.control.NumericEditField
        CollectFBGDataButton            matlab.ui.control.Button
        CaptureStereoImageButton        matlab.ui.control.Button
        CaptureEntireDataButton         matlab.ui.control.Button
        NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel  matlab.ui.control.Label
        RobotconnectionPanel            matlab.ui.container.Panel
        MotorsOnOffButton               matlab.ui.control.StateButton
        RobotDisconnectButton           matlab.ui.control.Button
        RobotConnectionLamp             matlab.ui.control.Lamp
        RobotConnectButton              matlab.ui.control.Button
        RobotIPEditField                matlab.ui.control.EditField
        RobotIPEditFieldLabel           matlab.ui.control.Label
    end

    
    properties (Access = private)
        galil_controller                % Galil Controller 
        sensor_id_run (1,:) {mustBeInteger} = [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]; % default 4 AA peak needle labeling
        Interrogator_t                  % Interrogator handle
        position_timer timer = timer(); % Timer to update current robot position
    end
    
    methods (Access = private)
        
        function UpdateDisplay(app, str)
            if isempty(app.DataStatusTextArea.Value{end})
                app.DataStatusTextArea.Value{end} = char(strrep(str, '\n', ''));
            else
                app.DataStatusTextArea.Value{end+1} = char(strrep(str, '\n', ''));
            end
        end
        
        function UpdateInsertionDepth(app)
            app.InsertionDepthEditField.Value = app.XAxisEditField.Value + app.LinearStageEditField.Value;
        end
        
        function UpdateRobotPosition(app)
            if ~isempty(app.galil_controller)
                positions = app.galil_controller.currentPosition();
                app.XAxisEditField.Value       = positions(1);
                app.YAxisEditField.Value       = positions(2);
                app.ZAxisEditField.Value       = positions(3);
                app.LinearStageEditField.Value = positions(4);
            end
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: RobotConnectButton
        function RobotConnectButtonPushed(app, event)
            % get IP Address
            ip_address = app.RobotIPEditField.Value;
            % connect to Galil Controller
            try
                app.galil_controller = GalilController(ip_address);
                app.galil_controller.waitCompletion = false;
                app.RobotConnectionLamp.Color = [0.0, 1.0, 0.0];
                
                pause(1e-3)
                app.galil_controller.allMotorsOff();
                
                % set position handles
                app.position_timer.Period = 0.1;
                app.position_timer.ExecutionMode = 'fixedRate';
                app.position_timer.TimerFcn = @(varargin) app.UpdateRobotPosition();
                start(app.position_timer);
                
            catch err
                disp(err.message);
                app.galil_controller = [];
                app.RobotConnectionLamp.Color = [1.0, 0.0, 0.0];
                stop(app.position_timer)
            end
            
        end

        % Button pushed function: RobotDisconnectButton
        function RobotDisconnectButtonPushed(app, event)
            pnet('closeall');
            clear app.galil_controller;
            app.galil_controller = [];
            app.RobotConnectionLamp.Color = [0.65, 0.65, 0.65];
            stop(app.position_timer)
        end

        % Value changed function: MotorsOnOffButton
        function MotorsOnOffButtonValueChanged(app, event)
            value = app.MotorsOnOffButton.Value;
            if ~isempty(app.galil_controller)
                if value
                    app.galil_controller.motorsOn(true, true, true, true)
                else
                    app.galil_controller.motorsOff(true, true, true, true)
                end
            else
                app.MotorsOnOffButton.Value = false;
            end

        end

        % Button pushed function: InterrogatorConnectButton
        function InterrogatorConnectButtonPushed(app, event)
            try
                ip_address = app.InterrogatorIPEditField.Value;
                app.Interrogator_t = FBGInterrogator(ip_address);
                app.InterrogatorConnectLamp.Color = [0, 1, 0];
            catch
                app.Interrogator_t = [];
                app.InterrogatorConnectLamp.Color = [1, 0, 0];
            end
            
            
        end

        % Button pushed function: InterrogatorDisconnectButton
        function InterrogatorDisconnectButtonPushed(app, event)
            clear app.Interrogator_t;
            app.Interrogator_t = [];
            app.InterrogatorConnectLamp.Color = [0.65, 0.65, 0.65];
        end

        % Value changed function: SensorIDRunEditField
        function SensorIDRunEditFieldValueChanged(app, event)
            value = app.SensorIDRunEditField.Value;
            temp_sensor_id_run = str2double(split(value, ','));
            if all(~isnan(temp_sensor_id_run))
                app.sensor_id_run = temp_sensor_id_run;
           
            end
            
            val = sprintf('%d,', app.sensor_id_run);
            val = strip(val, 'right', ',');
            app.SensorIDRunEditField.Value = val;
        end

        % Button pushed function: CollectFBGDataButton
        function CollectFBGDataButtonPushed(app, event)
            if isempty(app.Interrogator_t)
                return
            end
            
            % grab pertinent information
            N_samples = app.NumberFBGSamplesEditField.Value;
            num_of_sensors = length(app.sensor_id_run);
            
            % prepare data collection
            fbg_data = -1*ones(N_samples, num_of_sensors);
            
            % collect the FBG data
            sample_i = 0;
            while sample_i < N_samples
                sample_i = sample_i + 1; 
                
                % grab a row of FBG data sets
                try
                    [peaks, ~] = app.Interrogator_t.read_peaks();
                catch
                    continue
                end
                
                fbg_data(sample_i, :) = peaks; 
                
            end

            
            % save the data
            data_dir = fullfile(app.SaveDirectoryEditField.Value, ...
                                sprintf("Insertion%d",app.InsertionHoleSpinner.Value),...
                                num2str(app.InsertionDepthEditField.Value));
            mkdir(data_dir);
            fbgfile = fullfile(data_dir, "FBGdata.xls");
            
            writematrix(fbg_data, fbgfile);
            msg = sprintf("Saved FBG data to: %s\n", fbgfile);
            fprintf(msg);
            app.UpdateDisplay(msg);
            
        end

        % Button pushed function: CaptureStereoImageButton
        function CaptureStereoImageButtonPushed(app, event)
            pgr_stereo = py.PgrStereo.PgrStereo();
            pgr_stereo.connect();
            pgr_stereo.startCapture()
        
            % grab and convert the images to MATLAB format (rows x cols x [RGB])
            img_lr = cell(pgr_stereo.grab_image_pair());
            img_lr{1} = double(img_lr{1});
            img_lr{2} = double(img_lr{2});
        
            pgr_stereo.stopCapture()
            pgr_stereo.disconnect();
        
            Il = uint8(img_lr{1});
            Ir = uint8(img_lr{2});
            
            imshow(cat(2, Il, Ir), 'Parent',app.StereoImageAxes)
            
            % Save the data
            data_dir = fullfile(app.SaveDirectoryEditField.Value, ...
                                sprintf("Insertion%d",app.InsertionHoleSpinner.Value),...
                                num2str(app.InsertionDepthEditField.Value));
            mkdir(data_dir);
            
            left_file = fullfile(data_dir, 'left.png');
            right_file = fullfile(data_dir, 'right.png');
            
            imwrite(Il, left_file);
            msg = sprintf("Saved Image: %s\n", left_file);
            fprintf(msg);
            app.UpdateDisplay(msg);
            imwrite(Ir, right_file);
            msg = sprintf("Saved Image: %s\n", right_file);
            fprintf(msg);
            app.UpdateDisplay(msg);
            
        end

        % Button pushed function: CaptureEntireDataButton
        function CaptureEntireDataButtonPushed(app, event)
            CollectFBGDataButtonPushed(app,event);
            CaptureStereoImageButtonPushed(app,event);
            SaveRobotConfigButtonPushed(app, event);
        end

        % Selection changed function: MovementTypeButtonGroup
        function MovementTypeButtonGroupSelectionChanged(app, event)
            selectedButton = app.MovementTypeButtonGroup.SelectedObject;
            switch selectedButton
                case app.RelativeButton
                    % Change Movement Header Name
                    app.MovementLabel.Text = "Relative Movement (mm)";
                    
                    % reset the limits
                    app.XAxisMoveEditField.Limits       = [0, 50];
                    app.YAxisMoveEditField.Limits       = [0, 50];
                    app.ZAxisMoveEditField.Limits       = [0, 50];
                    app.LinearStageMoveEditField.Limits = [0, 50];
                    
                    % reset the movement values
                    app.XAxisMoveEditField.Value = 0;
                    app.YAxisMoveEditField.Value = 0;
                    app.ZAxisMoveEditField.Value = 0;
                    app.LinearStageMoveEditField.Value = 0;
                    
                    % Turn on movement direction
                    app.XAxisDirection.Enable = 'on';
                    app.YAxisDirection.Enable = 'on';
                    app.ZAxisDirection.Enable = 'on';
                    app.LSAxisDirection.Enable = 'on';
                    app.MovementDirectionLabel.Enable = 'on';
                    
                case app.AbsoluteButton
                    % Change Movement Header Name
                    app.MovementLabel.Text = "Absolute Movement (mm)";
                    
                    % reset the limits
                    app.XAxisMoveEditField.Limits       = [-inf, inf];
                    app.YAxisMoveEditField.Limits       = [-inf, inf];
                    app.ZAxisMoveEditField.Limits       = [-inf, inf];
                    app.LinearStageMoveEditField.Limits = [-inf, inf];
                    
                    % reset the movement values
                    app.XAxisMoveEditField.Value = app.XAxisEditField.Value;
                    app.YAxisMoveEditField.Value = app.YAxisEditField.Value;
                    app.ZAxisMoveEditField.Value = app.ZAxisEditField.Value;
                    app.LinearStageMoveEditField.Value = app.LinearStageEditField.Value;                    
                    
                    % Turn off movement direction
                    app.XAxisDirection.Enable = 'off';
                    app.YAxisDirection.Enable = 'off';
                    app.ZAxisDirection.Enable = 'off';
                    app.LSAxisDirection.Enable = 'off';
                    app.MovementDirectionLabel.Enable = 'off';
            end
            
            
            
        end

        % Button pushed function: ZeroAxesButton
        function ZeroAxesButtonPushed(app, event)
            app.XAxisEditField.Value = 0;
            app.YAxisEditField.Value = 0;
            app.ZAxisEditField.Value = 0;
            app.LinearStageEditField.Value = 0;
            
            if ~isempty(app.galil_controller)
                app.galil_controller.zeroAllAxes();
            end
        end

        % Button pushed function: MoveAllButton
        function MoveAllButtonPushed(app, event)
            if ~isempty(app.galil_controller)
                switch app.MovementTypeButtonGroup.SelectedObject
                    case app.RelativeButton
                        % grab values
                        x_movement = abs(app.XAxisMoveEditField.Value);
                        y_movement = abs(app.YAxisMoveEditField.Value);
                        z_movement = abs(app.ZAxisMoveEditField.Value);
                        ls_movement = abs(app.LinearStageMoveEditField.Value);
                        
                        % reset the values
                        app.XAxisMoveEditField.Value = x_movement;
                        app.YAxisMoveEditField.Value = y_movement;
                        app.ZAxisMoveEditField.Value = z_movement;
                        app.LinearStageMoveEditField.Value = ls_movement;
                        
                        % check for directions
                        if app.XAxisDirection.SelectedObject == app.XRetractButton
                            x_movement = -x_movement;
                        end
                        if app.YAxisDirection.SelectedObject == app.YAwayButton
                            y_movement = -y_movement;
                        end
                        if app.ZAxisDirection.SelectedObject == app.ZUpButton
                            z_movement = -z_movement;
                        end
                        if app.LSAxisDirection == app.LSInsertButton
                            ls_movement = -ls_movement;
                        end
                        
                        try
                            % move the robot
                            app.MoveAllButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveRelative(x_movement, y_movement, z_movement, ls_movement);
                            app.MoveAllButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveAllButton.BackgroundColor = [1, 0, 0];
                        end
                        
                    case app.AbsoluteButton
                        % grab current positions
                        x_current = app.XAxisEditField.Value;
                        y_current = app.YAxisEditField.Value;
                        z_current = app.ZAxisEditField.Value;
                        ls_current = app.LinearStageEditField.Value;
                        
                        % grab desired positions
                        x_desired = app.XAxisMoveEditField.Value;
                        y_desired = app.YAxisMoveEditField.Value;
                        z_desired = app.ZAxisMoveEditField.Value;
                        ls_desired = app.LinearStageMoveEditField.Value;
                        
                        % determine absolute movements
                        x_movement = x_desired;% - x_current;
                        y_movement = y_desired;% - y_current;
                        z_movement = z_desired;% - z_current;
                        ls_movement = ls_desired;% - ls_current;
                        
                        try
                            % move the robot
                            app.MoveAllButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveAbsolute(x_movement, y_movement, z_movement, ls_movement);
                            app.MoveAllButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveAllButton.BackgroundColor = [1, 0, 0];
                        end  
                end
            end
        end

        % Button pushed function: MoveXButton
        function MoveXButtonPushed(app, event)
            if ~isempty(app.galil_controller)
                
                switch app.MovementTypeButtonGroup.SelectedObject
                    case app.RelativeButton
                        % grab values
                        app.XAxisMoveEditField.Value = abs(app.XAxisMoveEditField.Value);
                        x_movement = abs(app.XAxisMoveEditField.Value);
                        if app.XAxisDirection.SelectedObject == app.XRetractButton % flip the sign
                            x_movement = -x_movement;
                        end
                        try
                            app.MoveXButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveRelative(x_movement, 0, 0, 0);
                            app.MoveXButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveXButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                                                
                    case app.AbsoluteButton
                        x_current = app.XAxisEditField.Value;
                        x_desired = app.XAxisMoveEditField.Value;
                        x_movement = x_desired;% - x_current;
                        try
                            app.MoveXButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveAbsolute(x_movement, 0, 0, 0);
                            app.MoveXButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveXButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                end
            end
        end

        % Button pushed function: MoveYButton
        function MoveYButtonPushed(app, event)
            if ~isempty(app.galil_controller)
                switch app.MovementTypeButtonGroup.SelectedObject
                    case app.RelativeButton
                        % grab values
                        app.YAxisMoveEditField.Value = abs(app.YAxisMoveEditField.Value);
                        y_movement = abs(app.YAxisMoveEditField.Value);
                        if app.YAxisDirection.SelectedObject == app.YTowardsButton
                            y_movement = -y_movement;
                        end
                        try
                            app.MoveYButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveRelative(0, y_movement, 0, 0);
                            app.MoveYButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveYButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                    case app.AbsoluteButton
                        y_current = app.YAxisEditField.Value;
                        y_desired = app.YAxisMoveEditField.Value;
                        y_movement = y_desired;% - y_current;
                        try
                            app.MoveYButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveAbsolute(0, y_movement, 0, 0);
                            app.MoveYButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveYButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                end
            end
        end

        % Button pushed function: MoveZButton
        function MoveZButtonPushed(app, event)
            if ~isempty(app.galil_controller)
                app.MoveZButton.BackgroundColor = [1.0, 1.0, 0.0];
                switch app.MovementTypeButtonGroup.SelectedObject
                    case app.RelativeButton
                        % grab values
                        app.ZAxisMoveEditField.Value = abs(app.ZAxisMoveEditField.Value); 
                        z_movement = abs(app.ZAxisMoveEditField.Value);
                        if app.ZAxisDirection.SelectedObject == app.ZUpButton
                            z_movement = -z_movement;
                        end
                        try
                            app.MoveZButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveRelative(0, 0, z_movement, 0);
                            app.MoveZButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveZButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                                               
                    case app.AbsoluteButton
                        z_current = app.ZAxisEditField.Value;
                        z_desired = app.ZAxisMoveEditField.Value;
                        z_movement = z_desired;% - z_current;
                        try
                            app.MoveZButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveAbsolute(0, 0, z_movement, 0);
                            app.MoveZButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveZButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                end
            end
        end

        % Button pushed function: MoveLSButton
        function MoveLSButtonPushed(app, event)
             if ~isempty(app.galil_controller)
                app.MoveLSButton.BackgroundColor = [1.0, 1.0, 0.0];
                switch app.MovementTypeButtonGroup.SelectedObject
                    case app.RelativeButton
                        % grab values
                        ls_movement = abs(app.LinearStageMoveEditField.Value);
                        app.LinearStageMoveEditField.Value = ls_movement;
                        if app.LSAxisDirection.SelectedObject == app.LSInsertButton
                            ls_movement = -ls_movement;
                        end                       
                        try
                            app.MoveLSButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveRelative(0, 0, 0, ls_movement);
                            app.MoveLSButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveLSButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                        
                    case app.AbsoluteButton
                        ls_current = app.LinearStageEditField.Value;
                        ls_desired = app.LinearStageMoveEditField.Value;
                        ls_movement = ls_desired;% - ls_current;
                        try
                            app.MoveLSButton.BackgroundColor = [1.0, 1.0, 0.0];
                            app.galil_controller.moveAbsolute(0, 0, 0, ls_movement);
                            app.MoveLSButton.BackgroundColor = [0.96, 0.96, 0.96];
                            
                        catch err
                            disp(err.message)
                            app.MoveLSButton.BackgroundColor = [1.0, 0.0, 0.0];
                        end
                end
             end
        end

        % Button pushed function: ClearStatusButton
        function ClearStatusButtonPushed(app, event)
            app.DataStatusTextArea.Value = {''};
        end

        % Button pushed function: ABORTButton
        function ABORTButtonPushed(app, event)
            if ~isempty(app.galil_controller)
                app.galil_controller.abort();
            end
        end

        % Button pushed function: SaveRobotConfigButton
        function SaveRobotConfigButtonPushed(app, event)
            if ~isempty(app.galil_controller) 
                pos = app.galil_controller.currentPosition();
                t = table(pos(1), pos(2), pos(3), pos(4), 'VariableNames', ...
                          {'X', 'Y', 'Z', 'LS'});
                data_dir = fullfile(app.SaveDirectoryEditField.Value, ...
                                sprintf("Insertion%d",app.InsertionHoleSpinner.Value),...
                                num2str(app.InsertionDepthEditField.Value));
                mkdir(data_dir);
                fileout = fullfile(data_dir, 'robot_config.xls');
                writetable(t, fileout);
                msg = sprintf("Saved robot configuration: %s\n", fileout);
                fprintf(msg);
                app.UpdateDisplay(msg);
            else
                disp("Not connected to robot to get current configuration.");
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1230 898];
            app.UIFigure.Name = 'MATLAB App';

            % Create RobotconnectionPanel
            app.RobotconnectionPanel = uipanel(app.UIFigure);
            app.RobotconnectionPanel.Title = 'Robot connection';
            app.RobotconnectionPanel.FontWeight = 'bold';
            app.RobotconnectionPanel.Position = [12 728 220 124];

            % Create RobotIPEditFieldLabel
            app.RobotIPEditFieldLabel = uilabel(app.RobotconnectionPanel);
            app.RobotIPEditFieldLabel.HorizontalAlignment = 'right';
            app.RobotIPEditFieldLabel.Position = [13 72 52 22];
            app.RobotIPEditFieldLabel.Text = 'Robot IP';

            % Create RobotIPEditField
            app.RobotIPEditField = uieditfield(app.RobotconnectionPanel, 'text');
            app.RobotIPEditField.Position = [80 72 100 22];
            app.RobotIPEditField.Value = '192.168.1.201';

            % Create RobotConnectButton
            app.RobotConnectButton = uibutton(app.RobotconnectionPanel, 'push');
            app.RobotConnectButton.ButtonPushedFcn = createCallbackFcn(app, @RobotConnectButtonPushed, true);
            app.RobotConnectButton.Position = [11 37 100 22];
            app.RobotConnectButton.Text = 'Connect';

            % Create RobotConnectionLamp
            app.RobotConnectionLamp = uilamp(app.RobotconnectionPanel);
            app.RobotConnectionLamp.Tag = 'ConnectionLamp';
            app.RobotConnectionLamp.Position = [191 73 20 20];
            app.RobotConnectionLamp.Color = [0.651 0.651 0.651];

            % Create RobotDisconnectButton
            app.RobotDisconnectButton = uibutton(app.RobotconnectionPanel, 'push');
            app.RobotDisconnectButton.ButtonPushedFcn = createCallbackFcn(app, @RobotDisconnectButtonPushed, true);
            app.RobotDisconnectButton.Position = [11 13 100 22];
            app.RobotDisconnectButton.Text = 'Disconnect';

            % Create MotorsOnOffButton
            app.MotorsOnOffButton = uibutton(app.RobotconnectionPanel, 'state');
            app.MotorsOnOffButton.ValueChangedFcn = createCallbackFcn(app, @MotorsOnOffButtonValueChanged, true);
            app.MotorsOnOffButton.Text = 'Motors On/Off';
            app.MotorsOnOffButton.Position = [118 13 93 22];

            % Create NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel = uilabel(app.UIFigure);
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel.FontSize = 24;
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel.FontWeight = 'bold';
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel.FontColor = [0 0.4471 0.7412];
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel.Position = [12 851 794 39];
            app.NeedleInsertion4AxisRobotStereoCameraInsertionExperimentLabel.Text = 'Needle Insertion 4-Axis Robot: Stereo Camera Insertion Experiment';

            % Create DataCollectionFBGandStereoCameraPanel
            app.DataCollectionFBGandStereoCameraPanel = uipanel(app.UIFigure);
            app.DataCollectionFBGandStereoCameraPanel.Title = 'Data Collection (FBG and Stereo Camera)';
            app.DataCollectionFBGandStereoCameraPanel.FontWeight = 'bold';
            app.DataCollectionFBGandStereoCameraPanel.Position = [258 647 963 201];

            % Create GridLayout2
            app.GridLayout2 = uigridlayout(app.DataCollectionFBGandStereoCameraPanel);
            app.GridLayout2.ColumnWidth = {118, 95, 84, 'fit', '1x', '1x'};
            app.GridLayout2.RowHeight = {22, 'fit', 22, 22, 22};
            app.GridLayout2.ColumnSpacing = 13.2857142857143;
            app.GridLayout2.Padding = [13.2857142857143 10 13.2857142857143 10];

            % Create CaptureEntireDataButton
            app.CaptureEntireDataButton = uibutton(app.GridLayout2, 'push');
            app.CaptureEntireDataButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureEntireDataButtonPushed, true);
            app.CaptureEntireDataButton.Layout.Row = 2;
            app.CaptureEntireDataButton.Layout.Column = 4;
            app.CaptureEntireDataButton.Text = 'Capture Entire Data';

            % Create CaptureStereoImageButton
            app.CaptureStereoImageButton = uibutton(app.GridLayout2, 'push');
            app.CaptureStereoImageButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureStereoImageButtonPushed, true);
            app.CaptureStereoImageButton.Layout.Row = 4;
            app.CaptureStereoImageButton.Layout.Column = 4;
            app.CaptureStereoImageButton.Text = 'Capture Stereo Image';

            % Create CollectFBGDataButton
            app.CollectFBGDataButton = uibutton(app.GridLayout2, 'push');
            app.CollectFBGDataButton.ButtonPushedFcn = createCallbackFcn(app, @CollectFBGDataButtonPushed, true);
            app.CollectFBGDataButton.Layout.Row = 3;
            app.CollectFBGDataButton.Layout.Column = 4;
            app.CollectFBGDataButton.Text = 'Collect FBG Data';

            % Create NumberFBGSamplesEditField
            app.NumberFBGSamplesEditField = uieditfield(app.GridLayout2, 'numeric');
            app.NumberFBGSamplesEditField.Layout.Row = 1;
            app.NumberFBGSamplesEditField.Layout.Column = 2;
            app.NumberFBGSamplesEditField.Value = 200;

            % Create FBGSamplesEditFieldLabel
            app.FBGSamplesEditFieldLabel = uilabel(app.GridLayout2);
            app.FBGSamplesEditFieldLabel.HorizontalAlignment = 'right';
            app.FBGSamplesEditFieldLabel.Layout.Row = 1;
            app.FBGSamplesEditFieldLabel.Layout.Column = 1;
            app.FBGSamplesEditFieldLabel.Text = '# FBG Samples';

            % Create SaveDirectoryEditField
            app.SaveDirectoryEditField = uieditfield(app.GridLayout2, 'text');
            app.SaveDirectoryEditField.Layout.Row = 1;
            app.SaveDirectoryEditField.Layout.Column = [4 5];

            % Create SaveDirectoryEditFieldLabel
            app.SaveDirectoryEditFieldLabel = uilabel(app.GridLayout2);
            app.SaveDirectoryEditFieldLabel.HorizontalAlignment = 'right';
            app.SaveDirectoryEditFieldLabel.Layout.Row = 1;
            app.SaveDirectoryEditFieldLabel.Layout.Column = 3;
            app.SaveDirectoryEditFieldLabel.Text = 'Save Directory';

            % Create InsertionHoleSpinner
            app.InsertionHoleSpinner = uispinner(app.GridLayout2);
            app.InsertionHoleSpinner.Limits = [1 10];
            app.InsertionHoleSpinner.Layout.Row = 2;
            app.InsertionHoleSpinner.Layout.Column = 2;
            app.InsertionHoleSpinner.Value = 1;

            % Create InsertionHoleSpinnerLabel
            app.InsertionHoleSpinnerLabel = uilabel(app.GridLayout2);
            app.InsertionHoleSpinnerLabel.HorizontalAlignment = 'right';
            app.InsertionHoleSpinnerLabel.Layout.Row = 2;
            app.InsertionHoleSpinnerLabel.Layout.Column = 1;
            app.InsertionHoleSpinnerLabel.Text = 'Insertion Hole';

            % Create InsertionDepthEditField
            app.InsertionDepthEditField = uieditfield(app.GridLayout2, 'numeric');
            app.InsertionDepthEditField.Layout.Row = 3;
            app.InsertionDepthEditField.Layout.Column = 2;

            % Create InsertionDepthmmEditFieldLabel
            app.InsertionDepthmmEditFieldLabel = uilabel(app.GridLayout2);
            app.InsertionDepthmmEditFieldLabel.HorizontalAlignment = 'right';
            app.InsertionDepthmmEditFieldLabel.Layout.Row = 3;
            app.InsertionDepthmmEditFieldLabel.Layout.Column = 1;
            app.InsertionDepthmmEditFieldLabel.Text = 'Insertion Depth (mm)';

            % Create SensorIDRunEditField
            app.SensorIDRunEditField = uieditfield(app.GridLayout2, 'text');
            app.SensorIDRunEditField.ValueChangedFcn = createCallbackFcn(app, @SensorIDRunEditFieldValueChanged, true);
            app.SensorIDRunEditField.Layout.Row = 5;
            app.SensorIDRunEditField.Layout.Column = [2 3];
            app.SensorIDRunEditField.Value = '1,2,3,4,5,6,7,8,9,10,11,12';

            % Create SensorIDRunEditFieldLabel
            app.SensorIDRunEditFieldLabel = uilabel(app.GridLayout2);
            app.SensorIDRunEditFieldLabel.HorizontalAlignment = 'right';
            app.SensorIDRunEditFieldLabel.Layout.Row = 5;
            app.SensorIDRunEditFieldLabel.Layout.Column = 1;
            app.SensorIDRunEditFieldLabel.Text = 'Sensor ID Run';

            % Create StatusTextAreaLabel
            app.StatusTextAreaLabel = uilabel(app.GridLayout2);
            app.StatusTextAreaLabel.HorizontalAlignment = 'center';
            app.StatusTextAreaLabel.FontWeight = 'bold';
            app.StatusTextAreaLabel.Layout.Row = 2;
            app.StatusTextAreaLabel.Layout.Column = [5 6];
            app.StatusTextAreaLabel.Text = 'Status';

            % Create DataStatusTextArea
            app.DataStatusTextArea = uitextarea(app.GridLayout2);
            app.DataStatusTextArea.Editable = 'off';
            app.DataStatusTextArea.Layout.Row = [3 5];
            app.DataStatusTextArea.Layout.Column = [5 6];

            % Create ClearStatusButton
            app.ClearStatusButton = uibutton(app.GridLayout2, 'push');
            app.ClearStatusButton.ButtonPushedFcn = createCallbackFcn(app, @ClearStatusButtonPushed, true);
            app.ClearStatusButton.Layout.Row = 5;
            app.ClearStatusButton.Layout.Column = 4;
            app.ClearStatusButton.Text = 'Clear Status';

            % Create InterrogatorConnectionPanel
            app.InterrogatorConnectionPanel = uipanel(app.UIFigure);
            app.InterrogatorConnectionPanel.Title = 'Interrogator Connection';
            app.InterrogatorConnectionPanel.FontWeight = 'bold';
            app.InterrogatorConnectionPanel.Position = [12 606 220 116];

            % Create InterrogatorConnectButton
            app.InterrogatorConnectButton = uibutton(app.InterrogatorConnectionPanel, 'push');
            app.InterrogatorConnectButton.ButtonPushedFcn = createCallbackFcn(app, @InterrogatorConnectButtonPushed, true);
            app.InterrogatorConnectButton.Position = [9 32 100 22];
            app.InterrogatorConnectButton.Text = 'Connect';

            % Create InterrogatorDisconnectButton
            app.InterrogatorDisconnectButton = uibutton(app.InterrogatorConnectionPanel, 'push');
            app.InterrogatorDisconnectButton.ButtonPushedFcn = createCallbackFcn(app, @InterrogatorDisconnectButtonPushed, true);
            app.InterrogatorDisconnectButton.Position = [9 11 100 22];
            app.InterrogatorDisconnectButton.Text = 'Disconnect';

            % Create InterrogatorConnectLamp
            app.InterrogatorConnectLamp = uilamp(app.InterrogatorConnectionPanel);
            app.InterrogatorConnectLamp.Position = [135 24 20 20];
            app.InterrogatorConnectLamp.Color = [0.651 0.651 0.651];

            % Create InterrgatorIPLabel
            app.InterrgatorIPLabel = uilabel(app.InterrogatorConnectionPanel);
            app.InterrgatorIPLabel.HorizontalAlignment = 'right';
            app.InterrgatorIPLabel.Position = [9 62 76 22];
            app.InterrgatorIPLabel.Text = 'Interrgator IP';

            % Create InterrogatorIPEditField
            app.InterrogatorIPEditField = uieditfield(app.InterrogatorConnectionPanel, 'text');
            app.InterrogatorIPEditField.Position = [100 62 111 22];
            app.InterrogatorIPEditField.Value = '192.168.1.11';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [83 24 1067 583];

            % Create RobotControlTab
            app.RobotControlTab = uitab(app.TabGroup);
            app.RobotControlTab.Title = 'Robot Control';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.RobotControlTab);
            app.GridLayout.ColumnWidth = {'1x', '1x', '1x', '1x', '1x'};
            app.GridLayout.RowHeight = {'1x', '1x', '1x', '1x', '1x', '1x'};

            % Create XAxisinsertionEditFieldLabel
            app.XAxisinsertionEditFieldLabel = uilabel(app.GridLayout);
            app.XAxisinsertionEditFieldLabel.HorizontalAlignment = 'center';
            app.XAxisinsertionEditFieldLabel.FontWeight = 'bold';
            app.XAxisinsertionEditFieldLabel.Layout.Row = 3;
            app.XAxisinsertionEditFieldLabel.Layout.Column = 1;
            app.XAxisinsertionEditFieldLabel.Text = {'X Axis'; '(+ insertion)'};

            % Create XAxisEditField
            app.XAxisEditField = uieditfield(app.GridLayout, 'numeric');
            app.XAxisEditField.Editable = 'off';
            app.XAxisEditField.HorizontalAlignment = 'center';
            app.XAxisEditField.Layout.Row = 3;
            app.XAxisEditField.Layout.Column = 2;

            % Create YAxistowardsyouLabel
            app.YAxistowardsyouLabel = uilabel(app.GridLayout);
            app.YAxistowardsyouLabel.HorizontalAlignment = 'center';
            app.YAxistowardsyouLabel.FontWeight = 'bold';
            app.YAxistowardsyouLabel.Layout.Row = 4;
            app.YAxistowardsyouLabel.Layout.Column = 1;
            app.YAxistowardsyouLabel.Text = {'Y Axis'; '(- towards you)'};

            % Create YAxisEditField
            app.YAxisEditField = uieditfield(app.GridLayout, 'numeric');
            app.YAxisEditField.Editable = 'off';
            app.YAxisEditField.HorizontalAlignment = 'center';
            app.YAxisEditField.Layout.Row = 4;
            app.YAxisEditField.Layout.Column = 2;

            % Create ZAxisdownEditFieldLabel
            app.ZAxisdownEditFieldLabel = uilabel(app.GridLayout);
            app.ZAxisdownEditFieldLabel.HorizontalAlignment = 'center';
            app.ZAxisdownEditFieldLabel.FontWeight = 'bold';
            app.ZAxisdownEditFieldLabel.Layout.Row = 5;
            app.ZAxisdownEditFieldLabel.Layout.Column = 1;
            app.ZAxisdownEditFieldLabel.Text = {'Z Axis'; '(+ down)'};

            % Create ZAxisEditField
            app.ZAxisEditField = uieditfield(app.GridLayout, 'numeric');
            app.ZAxisEditField.Editable = 'off';
            app.ZAxisEditField.HorizontalAlignment = 'center';
            app.ZAxisEditField.Layout.Row = 5;
            app.ZAxisEditField.Layout.Column = 2;

            % Create LinearStageinsertionEditFieldLabel
            app.LinearStageinsertionEditFieldLabel = uilabel(app.GridLayout);
            app.LinearStageinsertionEditFieldLabel.HorizontalAlignment = 'center';
            app.LinearStageinsertionEditFieldLabel.FontWeight = 'bold';
            app.LinearStageinsertionEditFieldLabel.Layout.Row = 6;
            app.LinearStageinsertionEditFieldLabel.Layout.Column = 1;
            app.LinearStageinsertionEditFieldLabel.Text = {'Linear Stage'; '(- insertion)'};

            % Create LinearStageEditField
            app.LinearStageEditField = uieditfield(app.GridLayout, 'numeric');
            app.LinearStageEditField.Editable = 'off';
            app.LinearStageEditField.HorizontalAlignment = 'center';
            app.LinearStageEditField.Layout.Row = 6;
            app.LinearStageEditField.Layout.Column = 2;

            % Create MoveXButton
            app.MoveXButton = uibutton(app.GridLayout, 'push');
            app.MoveXButton.ButtonPushedFcn = createCallbackFcn(app, @MoveXButtonPushed, true);
            app.MoveXButton.Layout.Row = 3;
            app.MoveXButton.Layout.Column = 5;
            app.MoveXButton.Text = 'Move X';

            % Create MoveYButton
            app.MoveYButton = uibutton(app.GridLayout, 'push');
            app.MoveYButton.ButtonPushedFcn = createCallbackFcn(app, @MoveYButtonPushed, true);
            app.MoveYButton.Layout.Row = 4;
            app.MoveYButton.Layout.Column = 5;
            app.MoveYButton.Text = 'Move Y';

            % Create MoveAllButton
            app.MoveAllButton = uibutton(app.GridLayout, 'push');
            app.MoveAllButton.ButtonPushedFcn = createCallbackFcn(app, @MoveAllButtonPushed, true);
            app.MoveAllButton.Layout.Row = 2;
            app.MoveAllButton.Layout.Column = 5;
            app.MoveAllButton.Text = 'Move All';

            % Create MoveZButton
            app.MoveZButton = uibutton(app.GridLayout, 'push');
            app.MoveZButton.ButtonPushedFcn = createCallbackFcn(app, @MoveZButtonPushed, true);
            app.MoveZButton.Layout.Row = 5;
            app.MoveZButton.Layout.Column = 5;
            app.MoveZButton.Text = 'Move Z';

            % Create MoveLSButton
            app.MoveLSButton = uibutton(app.GridLayout, 'push');
            app.MoveLSButton.ButtonPushedFcn = createCallbackFcn(app, @MoveLSButtonPushed, true);
            app.MoveLSButton.Layout.Row = 6;
            app.MoveLSButton.Layout.Column = 5;
            app.MoveLSButton.Text = 'Move LS';

            % Create RobotAxisLabel
            app.RobotAxisLabel = uilabel(app.GridLayout);
            app.RobotAxisLabel.HorizontalAlignment = 'center';
            app.RobotAxisLabel.FontWeight = 'bold';
            app.RobotAxisLabel.Layout.Row = 2;
            app.RobotAxisLabel.Layout.Column = 1;
            app.RobotAxisLabel.Text = 'Robot Axis';

            % Create CurrentPositionLabel
            app.CurrentPositionLabel = uilabel(app.GridLayout);
            app.CurrentPositionLabel.HorizontalAlignment = 'center';
            app.CurrentPositionLabel.FontWeight = 'bold';
            app.CurrentPositionLabel.Layout.Row = 2;
            app.CurrentPositionLabel.Layout.Column = 2;
            app.CurrentPositionLabel.Text = 'Current Position (mm)';

            % Create MovementTypeButtonGroup
            app.MovementTypeButtonGroup = uibuttongroup(app.GridLayout);
            app.MovementTypeButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @MovementTypeButtonGroupSelectionChanged, true);
            app.MovementTypeButtonGroup.TitlePosition = 'centertop';
            app.MovementTypeButtonGroup.Title = 'Movement Type';
            app.MovementTypeButtonGroup.Layout.Row = 1;
            app.MovementTypeButtonGroup.Layout.Column = [2 4];

            % Create RelativeButton
            app.RelativeButton = uiradiobutton(app.MovementTypeButtonGroup);
            app.RelativeButton.Text = 'Relative';
            app.RelativeButton.Position = [198 25 65 22];
            app.RelativeButton.Value = true;

            % Create AbsoluteButton
            app.AbsoluteButton = uiradiobutton(app.MovementTypeButtonGroup);
            app.AbsoluteButton.Text = 'Absolute';
            app.AbsoluteButton.Position = [373 25 69 22];

            % Create SaveRobotConfigButton
            app.SaveRobotConfigButton = uibutton(app.MovementTypeButtonGroup, 'push');
            app.SaveRobotConfigButton.ButtonPushedFcn = createCallbackFcn(app, @SaveRobotConfigButtonPushed, true);
            app.SaveRobotConfigButton.Position = [42 25 120 22];
            app.SaveRobotConfigButton.Text = 'Save Robot Config.';

            % Create XAxisMoveEditField
            app.XAxisMoveEditField = uieditfield(app.GridLayout, 'numeric');
            app.XAxisMoveEditField.Limits = [0 50];
            app.XAxisMoveEditField.HorizontalAlignment = 'center';
            app.XAxisMoveEditField.Layout.Row = 3;
            app.XAxisMoveEditField.Layout.Column = 4;

            % Create YAxisMoveEditField
            app.YAxisMoveEditField = uieditfield(app.GridLayout, 'numeric');
            app.YAxisMoveEditField.Limits = [0 50];
            app.YAxisMoveEditField.HorizontalAlignment = 'center';
            app.YAxisMoveEditField.Layout.Row = 4;
            app.YAxisMoveEditField.Layout.Column = 4;

            % Create ZAxisMoveEditField
            app.ZAxisMoveEditField = uieditfield(app.GridLayout, 'numeric');
            app.ZAxisMoveEditField.Limits = [0 50];
            app.ZAxisMoveEditField.HorizontalAlignment = 'center';
            app.ZAxisMoveEditField.Layout.Row = 5;
            app.ZAxisMoveEditField.Layout.Column = 4;

            % Create LinearStageMoveEditField
            app.LinearStageMoveEditField = uieditfield(app.GridLayout, 'numeric');
            app.LinearStageMoveEditField.Limits = [0 50];
            app.LinearStageMoveEditField.HorizontalAlignment = 'center';
            app.LinearStageMoveEditField.Layout.Row = 6;
            app.LinearStageMoveEditField.Layout.Column = 4;

            % Create MovementLabel
            app.MovementLabel = uilabel(app.GridLayout);
            app.MovementLabel.HorizontalAlignment = 'center';
            app.MovementLabel.FontWeight = 'bold';
            app.MovementLabel.Layout.Row = 2;
            app.MovementLabel.Layout.Column = 4;
            app.MovementLabel.Text = 'Relative Movement (mm)';

            % Create ZeroAxesButton
            app.ZeroAxesButton = uibutton(app.GridLayout, 'push');
            app.ZeroAxesButton.ButtonPushedFcn = createCallbackFcn(app, @ZeroAxesButtonPushed, true);
            app.ZeroAxesButton.BackgroundColor = [1 1 0];
            app.ZeroAxesButton.Layout.Row = 1;
            app.ZeroAxesButton.Layout.Column = 5;
            app.ZeroAxesButton.Text = 'Zero Axes';

            % Create XAxisDirection
            app.XAxisDirection = uibuttongroup(app.GridLayout);
            app.XAxisDirection.Layout.Row = 3;
            app.XAxisDirection.Layout.Column = 3;

            % Create XInsertButton
            app.XInsertButton = uiradiobutton(app.XAxisDirection);
            app.XInsertButton.Text = 'Insert';
            app.XInsertButton.Position = [39 47 58 22];
            app.XInsertButton.Value = true;

            % Create XRetractButton
            app.XRetractButton = uiradiobutton(app.XAxisDirection);
            app.XRetractButton.Text = 'Retract';
            app.XRetractButton.Position = [39 18 65 22];

            % Create YAxisDirection
            app.YAxisDirection = uibuttongroup(app.GridLayout);
            app.YAxisDirection.Layout.Row = 4;
            app.YAxisDirection.Layout.Column = 3;

            % Create YTowardsButton
            app.YTowardsButton = uiradiobutton(app.YAxisDirection);
            app.YTowardsButton.Text = 'Towards';
            app.YTowardsButton.Position = [39 47 67 22];
            app.YTowardsButton.Value = true;

            % Create YAwayButton
            app.YAwayButton = uiradiobutton(app.YAxisDirection);
            app.YAwayButton.Text = 'Away';
            app.YAwayButton.Position = [39 18 65 22];

            % Create ZAxisDirection
            app.ZAxisDirection = uibuttongroup(app.GridLayout);
            app.ZAxisDirection.Layout.Row = 5;
            app.ZAxisDirection.Layout.Column = 3;

            % Create ZUpButton
            app.ZUpButton = uiradiobutton(app.ZAxisDirection);
            app.ZUpButton.Text = 'Up';
            app.ZUpButton.Position = [39 47 37 22];
            app.ZUpButton.Value = true;

            % Create ZDowntButton
            app.ZDowntButton = uiradiobutton(app.ZAxisDirection);
            app.ZDowntButton.Text = 'Down';
            app.ZDowntButton.Position = [39 18 65 22];

            % Create LSAxisDirection
            app.LSAxisDirection = uibuttongroup(app.GridLayout);
            app.LSAxisDirection.Layout.Row = 6;
            app.LSAxisDirection.Layout.Column = 3;

            % Create LSInsertButton
            app.LSInsertButton = uiradiobutton(app.LSAxisDirection);
            app.LSInsertButton.Text = 'Insert';
            app.LSInsertButton.Position = [39 47 58 22];
            app.LSInsertButton.Value = true;

            % Create LSRetractButton
            app.LSRetractButton = uiradiobutton(app.LSAxisDirection);
            app.LSRetractButton.Text = 'Retract';
            app.LSRetractButton.Position = [39 18 65 22];

            % Create MovementDirectionLabel
            app.MovementDirectionLabel = uilabel(app.GridLayout);
            app.MovementDirectionLabel.HorizontalAlignment = 'center';
            app.MovementDirectionLabel.FontWeight = 'bold';
            app.MovementDirectionLabel.Layout.Row = 2;
            app.MovementDirectionLabel.Layout.Column = 3;
            app.MovementDirectionLabel.Text = 'Movement Direction';

            % Create ABORTButton
            app.ABORTButton = uibutton(app.GridLayout, 'push');
            app.ABORTButton.ButtonPushedFcn = createCallbackFcn(app, @ABORTButtonPushed, true);
            app.ABORTButton.BackgroundColor = [1 0 0];
            app.ABORTButton.FontSize = 26;
            app.ABORTButton.FontWeight = 'bold';
            app.ABORTButton.FontColor = [1 1 1];
            app.ABORTButton.Layout.Row = 1;
            app.ABORTButton.Layout.Column = 1;
            app.ABORTButton.Text = 'ABORT';

            % Create StereoImagesTab
            app.StereoImagesTab = uitab(app.TabGroup);
            app.StereoImagesTab.Title = 'Stereo Images';

            % Create StereoImageAxes
            app.StereoImageAxes = uiaxes(app.StereoImagesTab);
            title(app.StereoImageAxes, 'Left-Right Image Pair')
            zlabel(app.StereoImageAxes, 'Z')
            app.StereoImageAxes.Position = [0 -15 1046 562];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = NeedleInsertion_Galil_4Axes_app

            runningApp = getRunningApp(app);

            % Check for running singleton app
            if isempty(runningApp)

                % Create UIFigure and components
                createComponents(app)

                % Register the app with App Designer
                registerApp(app, app.UIFigure)
            else

                % Focus the running singleton app
                figure(runningApp.UIFigure)

                app = runningApp;
            end

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end