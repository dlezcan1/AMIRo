classdef FBGInterrogator < handle
    properties(SetAccess = protected)
        ip_address
        client      tcpclient
        port        uint16      = 1852; % micron sm130 default port (Dynamic)
    end
    properties (Access = public)
       timeout      double {mustBePositive} = 1.0; 
    end
    
    methods
        function obj = FBGInterrogator(ip_address, port, timeout)
           arguments
              ip_address;
              port          uint16 = 1852;
              timeout       double = 1.0;
           end
           obj.ip_address = ip_address;
           obj.port = port;
           obj.timeout = timeout;
           obj.connect();
        end
        
        % connect to the interrogator
        function connect(obj)
           obj.client =  tcpclient(obj.ip_address, obj.port, 'Timeout', obj.timeout,...
                            'ConnectTimeout', 5.0);
        end
        
        % get the peak data
        function [peaks,msg] = read_peaks(obj)
            obj.client.flush(); % fush the client output
%             obj.client.writeline('#GET_UNBUFFERED_DATA'); % prompt for unbuffered data
            obj.client.writeline('#GET_DATA'); % prompt for unbuffered data
            
            tic
            while obj.client.NumBytesAvailable < 10
                dt = toc;
                
                if dt > obj.client.Timeout
                    error("Read Timeout")
                end
            end
                
            msg_bytes = obj.client.read();
            
            
            if ~isempty(msg_bytes)
                msg = obj.parse_data_message(msg_bytes);
                peaks = cat(2, msg.signals{:});
            else
                msg = [];
                peaks = [];
            end
        end
    end
    
    methods(Static)
        function header = parse_data_header(header_bytes) 
           arguments
                header_bytes (1,:) uint8;
           end
           
           header_words = typecast(header_bytes, 'uint32');
           
           % grab channel signals detected
           header.serial_number     = char(typecast(header_words(8), 'int8'));
           header.CH_sigs           = double(typecast(header_words(5:6), 'int16'));
           header.granularity       = double(header_words(19));
           header.spectrum_start_wl = double(header_words(21))/header.granularity;
           header.spectrum_stop_wl  = double(header_words(22))/header.granularity;
           header.timestamp         = double(header_words(10)) + double(header_words(9))/1e6;
           header.error_code        = header_bytes(11*4 + 3);
           
        end
        
        function data = parse_data_message(msg_bytes)
            arguments
                msg_bytes (1,:) uint8;
            end
            
            data.msg_len = str2double(char(msg_bytes(1:10))); % length of the entire message
            data.header = FBGInterrogator.parse_data_header(msg_bytes(11:98));
            
            signals = double(typecast(msg_bytes(99:data.msg_len+10), 'int32'))/data.header.granularity;
            idx_0 = 1;
            for ch_i = 1:numel(data.header.CH_sigs)
                num_sig_i = data.header.CH_sigs(ch_i);
                
                data.signals{ch_i} = signals(idx_0:idx_0+num_sig_i-1);
                idx_0 = idx_0 + num_sig_i;
            end
        end
            
    end
    
end