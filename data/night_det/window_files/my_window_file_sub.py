import re
from skimage import io

#file=open('/media/hxw/d/moz/dataset/GT5576_change_gt2.txt')
file=open('/media/b3-542/LIBRARY/moz/dataset/GT5576_ClusterOnly.txt')
root_dir = '/media/b3-542/LIBRARY/moz/dataset/images/train/'
#root_dir2='/media/hxw/d/moz/dataset/bolb_without_xgt/'
lines=file.readlines()
result=open('train_subclass.txt','w')
result=open('train_subclass.txt','a+')
# result2=open('trainval.txt','w')
# result2=open('trainval.txt','a+')
index=0
file_format='jpg'
for line in lines:

    temp = re.split(' ', line)
    img_name=str(temp[0])
    car_num=int(temp[1])
    try:
        if car_num>0:
            img = io.imread('{0}{1}.{2}'.format(root_dir, img_name, file_format))
            result.write('# {0}\n'.format(index))
            # result2.write('{0}\n'.format(img_name))
            result.write('{0}{1}.{2}\n'.format(root_dir,img_name,file_format))
            #result.write('{0}{1}.{2}\n'.format(root_dir2, img_name, file_format))
            result.write('{0}\n{1}\n{2}\n{3}\n'.format(img.shape[2], img.shape[0], img.shape[1], car_num))


            for i in range(1,(car_num*5),5):
                label_i=int(temp[1+i])
                x_i=int(temp[2+i])
                y_i = int(temp[3 + i])
                w_i = int(temp[4 + i])
                h_i = int(temp[5 + i])
                if w_i|h_i>999:
                    print img_name
                #print [1,0,x_i,y_i,w_i,h_i]
                result.write('{0} 0 {1} {2} {3} {4}\n'.format(label_i,x_i,y_i,x_i+w_i,y_i+h_i))
            result.write('0\n')
            index=index+1
    except:
        pass