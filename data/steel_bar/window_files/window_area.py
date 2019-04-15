import re
from skimage import io
import random
import numpy as np

#file=open('/media/hxw/d/moz/dataset/GT5576_change_gt2.txt')
file=open('/media/b3-542/library2/moz/datasets/steelbar/train_labels.csv')
root_dir = '/media/b3-542/library2/moz/datasets/steelbar/train/'
# root_dir2='/media/hxw/d/moz/dataset/bolb_without_xgt/'
lines=file.readlines()

#random.shuffle(lines)
index=0
index_trainval=0
sum_area=0
areas=np.array([])
tmp_imgname='DBA90206.jpg'
tmp=np.array([[]])
for line in lines:

    temp = re.split(r'[,,\s]', line)
    #print temp
    img_name=str(temp[0])
    #car_num=int(temp[1])
    #try:
    if tmp_imgname!=img_name : #tmp_imgname!=img_name
        if index<125:
            img = io.imread('{0}{1}'.format(root_dir, tmp_imgname))
            #result.write('# {0}\n'.format(index_trainval))
            # result2.write('{0}\n'.format(img_name))
            #result.write('{0}{1}\n'.format(root_dir, tmp_imgname))
            # result.write('{0}{1}.{2}\n'.format(root_dir2, img_name, file_format))
            #result.write('{0}\n{1}\n{2}\n{3}\n'.format(img.shape[2], img.shape[0], img.shape[1], np.size(tmp, 0)))

            for i in range(0,np.size(tmp,0)):
                #print tmp
                x_i = int(tmp[i,0])
                y_i = int(tmp[i,1])
                xmax_i = int(tmp[i,2])
                ymax_i = int(tmp[i,3])

                sum_area=sum_area+(ymax_i-y_i)*(xmax_i-x_i)
                # print [1,0,x_i,y_i,w_i,h_i]
                #result.write('1 0 {0} {1} {2} {3}\n'.format(x_i, y_i, xmax_i, ymax_i))
            #result.write('0\n')
            areas=np.append(areas, [sum_area/(img.shape[0]* img.shape[1]*1.0)])
            print areas.mean()
            index_trainval = index_trainval + 1

        tmp = np.array([[int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])]])
        tmp_imgname = img_name
        tmp = np.array([[]])
        sum_area=0
        index=index+1


    tmp_new=np.array([[int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])]])

    if np.size(tmp,1)>0:
        tmp=np.concatenate([tmp,tmp_new])
    else:
        tmp=tmp_new
    #print tmp

        #index=index+1
    #except:
        #pass

