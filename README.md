# neural-network-from-scratch
Simple neural network to test what I've studied.
This network predicts below with 4 input nodes(x[0]~x[3]):
<br>x[0]+x[1] = y_test[0], x[2]+x[3] = y_test[1]

# Scheme
Simple Neural network with 3 layers - input, hidden, output.
Input layer has 4, hidden layer has #<i>h_nodes</i>, output layer has 2 nodes.
Error is defined by mean square error divided by 2.
## Setup

```python
def gen():
    global x, y_test
    x = np.random.randint(1, 200, 4)    #harder to predict as range expands
    y_test = np.array([x[0]+x[1], x[2]+x[3]])
#ideal output : o2[0] = x[0]+x[1], o2[1] = x[2]+x[3]

h_nodes = 10    #number of hidden nodes
w1 = np.random.randn(5, h_nodes)    #w1[4] is bias
net1 = []
o1 = []
w2 = np.random.randn(h_nodes+1, 2)      #w2[h_nodes] is bias
net2 = []
o2 = []     #prediction - compare with y_test
```
Final prediction result is o2.

```python
#leaky relu
def myRelu(x):
    if x<0:
        return -0.01*x
    else:
        return x
def dMyRelu(x):
    if x<0:
        return -0.01
    else:
        return 1

def forward():
    global net1, o1, net2, o2
    net1 = np.matmul(x, w1[:len(w1)-1])
    for i in range(len(net1)):
        net1[i]+=w1[len(w1)-1][i]
    o1 = net1
    for i in range(len(o1)):
        o1[i] = myRelu(o1[i])
    net2 = np.matmul(o1, w2[:len(w2)-1])
    for i in range(len(net2)):
        net2[i] += w2[len(w2)-1][i]
    o2 = net2
    for i in range(len(o2)):
        o2[i] = myRelu(o2[i])

def backProp():   #back propagation
    for i in range(len(w2)):    #ith node
        for j in range(len(w2[i])): #jth link
            if i==(len(w2)-1):  #bias
                w2[i][j] += lr*dMyRelu(net2[j])*(y_test[j]-o2[j])
            else:
                w2[i][j] += lr*o1[i]*dMyRelu(net2[j])*(y_test[j] - o2[j])
    forward()
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            if i==(len(w1)-1):
                tmpsum = 0
                for k in range(len(w2[j])):
                    tmpsum += w2[j][k]*dMyRelu(net2[k])*(y_test[k] - o2[k])
                w1[i][j] += lr*dMyRelu(net1[j])*tmpsum
                    
            else:
                tmpsum = 0
                for k in range(len(w2[j])):
                    tmpsum += w2[j][k]*dMyRelu(net2[k])*(y_test[k] - o2[k])
                w1[i][j] += lr*x[i]*dMyRelu(net1[j])*tmpsum

def train():
    gen()
    forward()
    backProp()

def trains(n):
    for i in range(n):
        train()
```
# Result
<div>
  <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/step10_end3000.png">
</div>
You can set batch size and epoch by variable <i>step</i> and <i>end</i>. In this example overfitting has occured around 2700 epoch.
