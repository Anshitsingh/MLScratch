
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


x_train=mnist.train.images.reshape(55000,28,28)
y_train=mnist.train.labels
print y_train.shape


# In[3]:


image=x_train[:50][:][:]


# In[4]:


f=np.random.rand(3,3)
out=np.zeros(shape=(image.shape[0],image.shape[1]))
i_h=image.shape[1]
i_w=image.shape[2]
d=image.shape[0]
f_h=f.shape[0]
f_w=f.shape[1]
p_h=1
p_w=1
output_list=[]

s_w=2
s_h=2




# In[5]:


#i tried vector form a[:][p+i][p+j]=image[:][i][j] but it was showing some error
def padding(image,a):
    
    p=1
    for k in range(image.shape[0]):
        for i in range(image.shape[1]):
                for j in range(image.shape[2]):
                    a[k][p+i][p+j]=image[k][i][j]

        
            
    return a
                


# In[6]:


def convolve(image,fil):
    out=[]
    pad=1
    
    i_h=image.shape[1]
    i_w=image.shape[2]
    s_h=1
    s_w=1
    f_h=3
    f_w=3  
    k=0
    l=i_h-f_h+1
    y=0
    u=0
    
    m=image.shape[0]
    f=fil
    k=1
    count=0
    #this is the required output image and that is 2 less than the padded image or equal to the before padded image
    x=image.shape[1]-2
    y=image.shape[2]-2
    prod=x*y
    conv=np.zeros(shape=(image.shape[0],3,3))
    out=np.zeros(shape=(image.shape[0],x,y))
    for k in range(m):
        a=image[k]
        for i in range(i_h-f_h+1):
            for j in range(i_w-f_w+1):
                        
                summation=a[i:i+3,j:j+3]
                
                conv=summation*f
                         
                convolve=np.sum(conv)
                
                out[k][i][j]=convolve
                
    
    return out                     
                               
           
        
    


# In[7]:


def sigmoid(image):
    out=1/(1+np.exp(-1*image))
    return out


# In[8]:


def relu(image):
    
    image[image<0]=0
    return image
    


# In[9]:


def maxpool(image):
    s=1
    f=2
    out=np.zeros(shape=(image.shape[0],image.shape[1]-1,image.shape[2]-1))
    
    for k in range(image.shape[0]):
        for i in range(image.shape[1]-1):
            for j in range(image.shape[2]-1):
                maximum=0
                for p in range(2):
                    for q in range(2):
                        if(image[k][i+p][j+q]>maximum):
                            maximum=image[k][i+p][j+q]
                out[k][i][j]=maximum
                
        
    return out


# In[10]:


#sigmoid layer

def forwardlayer(f,image):
    #using formula to calculate padding
    pad=1
    x=1*2+image.shape[1]
    a=np.zeros(shape=(image.shape[0],x,x))
    a=padding(image,a)
    
    conv=convolve(a,f)
    
       
    
    relu_output=relu(conv)
    

    
    maxpoolout=maxpool(relu_output)
    
    return maxpoolout



# In[11]:


maxpool_out1=forwardlayer(f,image)
print "maxpool_out after layer 1 shape is ",maxpool_out1.shape
maxpool_out2=maxpool_out1


# In[12]:


product=maxpool_out2.shape[1]*maxpool_out2.shape[2]


# In[13]:


final_input=maxpool_out2.reshape(maxpool_out2.shape[0],product)
fc_weights=np.random.randn(10,product)
final_output=np.dot(fc_weights,final_input.T)
final_output=final_output.T
print final_output.shape


# In[14]:


def softmax(final_output):
    softmax_in=np.exp(final_output)
    softmax_output=np.zeros(final_output.shape)
    denominator=np.sum(softmax_in,axis=0)
    softmax_output=softmax_in/denominator
    
    return softmax_output
    


# In[15]:


def softmax_classifier(softmax_output):
    predict=np.zeros(shape=(softmax_output.shape[0],1))
    pred=np.argmax(softmax_output,axis=1)
    
    predict=pred.reshape(len(pred),1)
    predicted=np.zeros(shape=(predict.shape[0],10))
    m=predict.shape[0]
    for i in range(m):
        for j in range(10):
            if(predict[i][0]==j):
                predicted[i][j]=1
        
    
    
    return predicted


# In[16]:


classifier_input=softmax(final_output)


# In[17]:


predicted=softmax_classifier(classifier_input)


# In[18]:


print predicted


# In[19]:


#starting to write functions for backprop ----forward prop done


# In[20]:


def final_loss(target,classifier_input):
    
    predicted=classifier_input
    diff=np.subtract(target,predicted)
    m=target.shape[0]
    
    #softmax loss summed over all classes
    loss=((-np.dot(target,(np.log(predicted.T+10e-9)))))/m
                      
    fin_loss=np.zeros(shape=(m,1))
    
    for k in range(m):
        fin_loss[k]=loss[k][k]
    #do sum
    fin_loss_total=np.sum(fin_loss)/m
    return fin_loss_total

        
    
    

    


# In[21]:


target=y_train[:50][:][:]
print "predicted shape",predicted.shape


loss=final_loss(target,predicted)
print loss


# In[22]:


def softmaxlossgradient(target,predicted,weights,classifier_input):
    w_width=weights.shape[1]
    m=len(target)
    gradient=np.zeros(shape=(m,10))
    gradient=np.multiply(predicted,(target-predicted))
    
    return gradient
                    

            


# In[23]:


#gradient wrt fc layer weights

def softmaxderivativewrtw(target,predicted,weights,classifier_input,x):
    
    #size of returned is mX10 i.e softmaxgradpower
    #we want all weights gradient--grad--so shape is 10Xweights.shape[2]
    #x shape is mXweights.shape[2]
    
    w_width=weights.shape[1]
    softmaxgradwrtpower=softmaxlossgradient(target,predicted,weights,classifier_input)
    
    gradw=np.zeros(shape=(weights.shape))
    product=x.shape[1]*x.shape[2]
    a=x.reshape(target.shape[0],product)
    gradw=np.matmul(softmaxgradwrtpower.T,a)
    return gradw
           
            
    


# In[24]:


#gradient wrt input image

def softmaxderivativewrtx(target,predicted,weights,classifier_input,x):
    
    #size of returned is mX10 i.e softmaxgradpower
    #we want all weights gradient--grad--so shape is 10Xweights.shape[2]
    #x shape is mXweights.shape[2]
    
    m=len(target)
    w_width=weights.shape[1]
    softmaxgradwrtpower=softmaxlossgradient(target,predicted,weights,classifier_input) 
    gradx=np.zeros(shape=(x.shape))
    
    gradx=np.matmul(softmaxgradwrtpower,weights)
    a=math.sqrt(gradx.shape[1])
    a=int(a)
                
    gradx=gradx.reshape(gradx.shape[0],a,a)
                
    
    return gradx
            
                
            
    


# In[25]:


def maxpoolderivative(convoutput,gradx,image):
    s=1
    f=2
    largerderivative=np.zeros(shape=(image.shape))
    
    for k in range(image.shape[0]):
        for i in range(image.shape[1]-1):
            for j in range(image.shape[2]-1):
                
                maximum=0
                alpha=i
                beta=j
                for p in range(2):
                    for q in range(2):
                        if(image[k][i+p][j+q]>maximum):
                            maximum=image[k][i+p][j+q]
                            alpha=i+p
                            beta=j+q
               
                largerderivative[k][alpha][beta]=largerderivative[k][alpha][beta]+gradx[k][i][j]
    
    return largerderivative
    


# In[ ]:


#remember that the output here is the before maxpool output and not the after maxpool output
#(outputfrom convolution,image,fc_weights,largerderivative,k)

def convandreluderivativewrtw(output,x,weights,gradx,k):
    m=x.shape[0]    
    gradconvweights=np.zeros(shape=(convweights.shape[0],convweights.shape[1]))
    
    h=output.shape[1]
    w=output.shape[2]
    
    for i in range(h):
            for j in range(w):
                for p in range(3):
                    for q in range(3):
                        if(output[k][i][j]>0):
                            gradconvweights[p][q]=gradconvweights[p][q]+gradx[k][i][j]*x[k][i+p][j+q]                
    
    return gradconvweights
    
#output shape is  (3, 28, 28)
#gradx[largerderivative is sent here] shape is (3, 28, 28)
#x shape is  (3, 28, 28)    


# In[ ]:


#starting backprop using above functions

total_loss=final_loss(target,predicted)
m=target.shape[0]
w_width=fc_weights.shape[1]
alpha=1000
alpha2=10
numiterations=10
convweights=f
for e in range(numiterations):
        
    delta=1e-1
    if(loss>delta):
        for k in range(m):
            
            x=forwardlayer(convweights,image)
            
            softmaxgradwrtpower=softmaxlossgradient(target,predicted,fc_weights,classifier_input)
            #updating fc weightstarget,predicted,fc_weights,classifier_input) 
            gradw=softmaxderivativewrtw(target,predicted,fc_weights,classifier_input,x)
            #updating fc weights
            gradx=softmaxderivativewrtx(target,predicted,fc_weights,classifier_input,x)
            x=1*2+image.shape[1]
            a=np.zeros(shape=(image.shape[0],x,x))
            a=padding(image,a)   
            output=convolve(a,convweights)
            
            largerderivative=maxpoolderivative(output,gradx,image)
            
            convweightsgrad=convandreluderivativewrtw(output,a,fc_weights,largerderivative,k)
            
            #updating convweights
            
            convweights=convweights-alpha2*convweightsgrad/m 
            
            maxpool_out=forwardlayer(convweights,image)
            #fc layer
            fc_weights=fc_weights-(alpha*gradw)/m
            print "fc_weights gradients iteration",e,"example",k,"are",gradw
            final_input=maxpool_out.reshape(maxpool_out.shape[0],product)
            final_output=np.dot(fc_weights,final_input.T)
            final_output=final_output.T
            #softmax
            classifier_input=softmax(final_output)
            predicted=softmax_classifier(classifier_input)
            loss=final_loss(target,predicted)        
            print "loss after iteration",e,"is",loss                
                                      
        if(e>5):
            alpha=alpha/2 
            alpha2=alpha2/2
            

