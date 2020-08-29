# neural-network-from-scratch
Simple neural network scheme to compare the performance of NNs with different number of hidden nodes and learning rate.
This network predicts below with 4 input nodes(x[0]~x[3]):
<br>x[0]+x[1] = y_test[0], x[2]+x[3] = y_test[1]

## Scheme
Simple neural network with 3 layers - input, hidden, output.
Number of nodes of input layer is 4, output layer is 2 and hidden layer is variable <i>h_nodes</i>.
Error is defined by mean squared error.
Default range of x[k]: 1~200
Default learning rate: lr = 0.00001

## FYI
This network does not normalize inputs. Therefore wide range of x, large lr, large number of hidden nodes can easily trigger overflow which leads to exception.

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
Here batch size is set to 20 and epoch is 250.
<div>
  <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node2.png">
</div>
Above is plotted with two hidden nodes. (<i>h_nodes</i> = 2) Performance is quite poor.
<br></br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node3.png">
</div>
Here <i>h_nodes</i> = 3. Performance got better but still not so good.
<br></br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node4.png">
</div>
<i>h_nodes</i> = 4. Overfitting occured around 75(# of train=1500) epoch.
<br></br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node6.png">
</div>
<i>h_nodes</i> = 6.
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node8.png">
</div>
<i>h_nodes</i> = 8. Shows good performance around 100 epoch but soon exhibits overfitting after that.
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node10.png">
</div>
<i>h_nodes</i> = 10. Overfitting occured after 150 epoch.

# More result with lower learning rate
<i>lr</i> = 0.00001 was too big for NN with number of hidden nodes exceeding 10 - tests were flooded with overflows :(
<br>So here I decided to lower it to 0.000001.</br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node2_lr000001.png">
</div>
h_nodes = 2.
<br></br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node6_lr000001.png">
</div>
h_nodes = 6.
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node10_lr000001_3.png">
</div>
h_nodes = 10. Consistently displayed nice learning curve.
<br></br>
<div>
    <img src="https://github.com/ferrarijh/neural-network-from-scratch/blob/master/demo/node20_lr000001.png">
</div>
<i>h_nodes</i> = 20. Best performance so far. Tested multiple times and this setup consistently showed superb result.
<br></br>

# Conclusion
Bigger number of hidden nodes doesn't necessarily lead to higher performance. However, moderately bigger number of it can help with lower learning rate.
