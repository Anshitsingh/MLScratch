
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import math
image= cv2.imread("/home/jash/Desktop/dl/rubix.png")


# In[2]:


f=np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,0,1],[-2,0,2],[-1,0,1]]])
out=np.zeros(shape=(image.shape[0],image.shape[1]))
i_h=image.shape[0]
i_w=image.shape[1]
d=image.shape[2]
f_h=f.shape[0]
f_w=f.shape[1]
p_h=1
p_w=1
output_list=[]
s_w=2
s_h=2

#using formula to calculate padding
pad=1
x=1*2+225
a=np.zeros(shape=(x,x,3))


# In[3]:


def padding(image,a):
    p=1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                a[p+i][p+j][k]=image[i][j][k]
    return a
                


# In[4]:


a=padding(image,a)
print a.shape


# In[5]:


def convolve(image,fil):
    out=[]
    pad=1
    print image.shape
    i_h=image.shape[0]
    i_w=image.shape[1]
    s_h=1
    s_w=1
    d=3
    l=i_h-f_h+1
    y=0
    print "a shape is",a.shape
    print "l is",l
    for i in range(i_h-f_h+1):
        
        
        for j in range(i_w-f_w+1):
            sum=0
            for k in range(d):
                for p in range(3):
                    for q in range(3):
                        sum=sum+image[i+p][j+q][k]*f[p][q][k]
                        
            out.append(sum)
        
    print len(out)/225
    return (out)


# In[6]:


def convert(sum,out,image):
    print "image shape is ",image.shape
    k=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i][j]=sum[k]
            k=k+1
    return out    


# In[7]:


def sigmoid(image):
    out=1/(1+np.exp(-1*image))
    return out


# In[8]:



def maxpool(image):
    s=2
    f=2
    out=[]
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            maximum=0
            for p in range(2):
                for q in range(2):
                    if(image[i+p][j+q]>maximum):
                        maximum=image[i+p][j+q]
            out.append(maximum)
            j=j+1
        i=i+1
    return out


# In[9]:



def convertmaxpool(maxpool_list,image):
    out=np.zeros(shape=( (image.shape[0]-1), (image.shape[1]-1) ))
    
    k=0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j]=maxpool_list[k]
            k=k+1
    return out


# In[ ]:


def forwardlayer1(a,f,image):
    output_list=convolve(a,f)
    out=np.zeros(shape=(image.shape[0],image.shape[1]))  
    out=convert(output_list,out,image)
    sigmoid_output=sigmoid(out)
    maxpool_list=[]
    maxpool_list=maxpool(sigmoid_output)
    maxpool_out=convertmaxpool(maxpool_list,sigmoid_output)
    return maxpool_out



# In[ ]:


maxpool_out=forwardlayer1(a,f,image)
print maxpool_out.shape
cv2.imshow("maxpooled output",maxpool_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

