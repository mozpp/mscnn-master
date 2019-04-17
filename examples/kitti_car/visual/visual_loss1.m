clear all; close all; clc;
log_file = '../mobile_mscnn_384/log_2nd.txt';%mobile_mscnn_384
fid = fopen(log_file, 'r');
%fid_accuracy = fopen('/home/wangxiao/Downloads/output_accuracy.txt', 'w'); 
%fid_loss = fopen('/home/wangxiao/Downloads/output_loss.txt', 'w');

iteration ={};
loss = [];
item={'#0: accuracy_1_5x5 = ','#1: accuracy_1_5x5 = ',...
    '#2: accuracy_1_7x7 = ','#3: accuracy_1_7x7 = ',...
    '#4: accuracy_2_5x5 = ','#5: accuracy_2_5x5 = ',...
    '#6: accuracy_2_7x7 = ','#7: accuracy_2_7x7 = ',...
    '#8: accuracy_3_5x5 = ','#9: accuracy_3_5x5 = ',...
    '#10: accuracy_3_7x7 = ','#11: accuracy_3_7x7 = ',...
    '#12: accuracy_4_5x5 = ','#13: accuracy_4_5x5 = '};
res=cell(length(item),1);
accuracy = {};
% path = '/home/wangxiao/Downloads/';
% fid_ = fopen([path, 'loss_file_.txt'], 'a');
while(~feof(fid))
    tline = fgetl(fid);
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
    for i=1:length(item)
    if strfind(tline, 'solver.cpp:244]')
        iter_index = strfind(tline, item{1,i});
        loss_current = str2num(tline((iter_index+21):end));
        %         fprintf(fid_, '%s \n', loss_current );
        if(loss_current~=-1)
            loss_current=(sum(res{i})+loss_current)/(length(res{i})+1);
            res{i} = [res{i}  loss_current] ;       % count the iteration;
        end
    end
    end
end

res1=[];
for i=1:length(item)
    plot(res{i});
    hold on;
    res1=[res1;res{i}(length(res{i}))];
end
legend(item);
res1
% plot(boxiou_1_5x5);hold on;plot(boxiou_1_7x7);plot(boxiou_2_5x5);
% plot(boxiou_2_7x7);plot(boxiou_3_5x5);plot(boxiou_3_7x7);plot(boxiou_4_5x5);