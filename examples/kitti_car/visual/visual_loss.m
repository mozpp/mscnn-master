clear all; close all; clc;
log_file = '../mscnn_ori/log_2nd.txt';%mobile_mscnn_384
fid = fopen(log_file, 'r');
%fid_accuracy = fopen('/home/wangxiao/Downloads/output_accuracy.txt', 'w'); 
%fid_loss = fopen('/home/wangxiao/Downloads/output_loss.txt', 'w');

iteration ={};
loss = [];
boxiou_1_5x5=[];boxiou_1_7x7=[];boxiou_2_5x5=[];
boxiou_2_7x7=[];boxiou_3_5x5=[];boxiou_3_7x7=[];boxiou_4_5x5=[];
accuracy = {};
% path = '/home/wangxiao/Downloads/';
% fid_ = fopen([path, 'loss_file_.txt'], 'a');
while(~feof(fid))
    tline = fgetl(fid);
    %%
    if strfind(tline, 'sgd_solver.cpp:')
        iter_index = strfind(tline, 'Iteration ');
        rest = tline((iter_index+9):end);
        iter_current = strtok(rest, ',');                   % iteration number;
        iteration = [iteration  iter_current];        % count the iteration; 
        lr_index = strfind(tline, 'lr = ');
        lr_current = tline((lr_index+4):end);                  % learning rate;
    end
    
    %%
    if strfind(tline, 'solver.cpp:228]')
        iter_index = strfind(tline, 'loss = ');
        loss_current = str2num(tline((iter_index+7):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(loss)+loss_current)/(length(loss)+1);
            loss = [loss  loss_current] ;       % count the iteration;
        end
    end
    %%
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_1_5x5 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_1_5x5)+loss_current)/(length(boxiou_1_5x5)+1);
            boxiou_1_5x5 = [boxiou_1_5x5  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_1_7x7 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_1_7x7)+loss_current)/(length(boxiou_1_7x7)+1);
            boxiou_1_7x7 = [boxiou_1_7x7  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_2_5x5 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_2_5x5)+loss_current)/(length(boxiou_2_5x5)+1);
            boxiou_2_5x5 = [boxiou_2_5x5  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_2_7x7 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_2_7x7)+loss_current)/(length(boxiou_2_7x7)+1);
            boxiou_2_7x7 = [boxiou_2_7x7  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_3_5x5 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_3_5x5)+loss_current)/(length(boxiou_3_5x5)+1);
            boxiou_3_5x5 = [boxiou_3_5x5  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_3_7x7 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_3_7x7)+loss_current)/(length(boxiou_3_7x7)+1);
            boxiou_3_7x7 = [boxiou_3_7x7  loss_current] ;       % count the iteration;
        end
    end
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, 'boxiou_4_5x5 = ');
        loss_current = str2num(tline((iter_index+15):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(boxiou_4_5x5)+loss_current)/(length(boxiou_4_5x5)+1);
            boxiou_4_5x5 = [boxiou_4_5x5  loss_current] ;       % count the iteration;
        end
    end
    
%     if strfind(tline, 'aver_accuracy: ')
%         aver_accuracy_index = strfind(tline, 'aver_accuracy: ');
%         aver_accuracy_current = tline((aver_accuracy_index+15):end);
%        
%         accuracy = [accuracy  aver_accuracy_current];
%     end
end
    

%loss_file_Path = importdata('/home/wangxiao/Downloads/loss_file_.txt');
%plot(loss);legend('loss');hold on;
plot(boxiou_1_5x5);hold on;plot(boxiou_1_7x7);plot(boxiou_2_5x5);
plot(boxiou_2_7x7);plot(boxiou_3_5x5);plot(boxiou_3_7x7);plot(boxiou_4_5x5);