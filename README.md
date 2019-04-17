
# Caffe LSTM Example on Sin(t) Waveform Prediction
- Network & Solver definition
- Mini-Batch Training & Testing
- Inifnite time sequence inference
- Iterative Finetuning

Example based on http://www.xiaoliangbai.com/2018/01/30/Caffe-LSTM-SinWaveform-Batch by Xiaoliang Bai.<br>



```python
import numpy as np
import math
import os
import caffe
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# change this to use CPU/GPU acceleration
USE_GPU = 0

if USE_GPU:
    GPU_ID = 0
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
else:
    caffe.set_mode_cpu()

%load_ext autoreload
%autoreload 2
```


```python
# Use the sample generator from Tensorflow Sin(t) online
def generate_sample(f = 1.0, t0 = None, batch_size = 1, predict = 50, samples = 100):
    """
    Generates data samples.

    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100.0

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, samples + predict) / Fs
        if _t0 is None:
            t0 = np.random.rand() * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        freq = f
        if freq is None:
            freq = np.random.rand() * 3.5 + 0.5

        y = np.sin(2 * np.pi * freq * (t + t0))

        T[i, :] = t[0:samples]
        Y[i, :] = y[0:samples]

        FT[i, :] = t[samples:samples + predict]
        FY[i, :] = y[samples:samples + predict]

    return T, Y, FT, FY
```


```python
net_path = 'lstm_demo_network.train.prototxt'
inference_path = 'lstm_demo_network.deploy.prototxt'
solver_config_path = 'lstm_demo_solver.prototxt'
snapshot_prefix = 'lstm_demo_snapshot'
# Network Parameters
n_input = 1 # single input stream
n_steps = 100 # timesteps
n_hidden = 15 # hidden units in LSTM
n_outputs = 50 # predictions in future time
batch_size = 20 # batch of data
NO_INPUT_DATA = -2 # defined numeric value for network if no input data is available
# Training Parameters
n_train = 4000
n_display = 200
n_adamAlpha = 0.002
```

## Training Network definition


```python
#define and save the network
from caffe import layers as L, params as P
def gen_network(n_steps,n_outputs,batch_size,n_input,n_hidden):
    n = caffe.NetSpec()
    # the network is trained on a time series of n_steps + n_outputs
    
    # we have input data for n_steps, which the network needs to keep tracing for n_output more steps
    # labels are given for n_output steps for loss calculation
    
    # a third input (clip) gives a mask for the first element in each time series.
    # this seems cumbersome but it allows training on series longer than the unrolled network size
    # as well as inference on infinite sequences
    
    # the data shape for caffe LSTM/RNN layers is always T x B x ... where T is the timestep and B are
    # independent data series (for example multiple batches)
    n.data, n.label, n.clip = L.Input( shape=[ dict(dim=[n_steps,batch_size,n_input]),
                                           dict(dim=[n_outputs,batch_size,n_input]),
                                           dict(dim=[n_steps+n_outputs,batch_size]) ],ntop=3)

    # the data layer size must match the size of the unrolled network, as such we concatenate with dummy-values
    # indicating that no more input data is available beyond n_steps
    n.dummy=L.DummyData( data_filler=dict(type="constant",value = NO_INPUT_DATA),
                        shape=[ dict(dim=[n_outputs,batch_size,n_input]) ] )
    n.fulldata=L.Concat(n.data, n.dummy, axis=0)
    # the lstm layer with n_hidden neurons
    n.lstm1 = L.LSTM(n.fulldata, n.clip,
                     recurrent_param = dict(num_output=n_hidden,
                                            weight_filler = dict(type='gaussian', std=0.05),
                                            bias_filler = dict(type='constant',value=0)))
    # usually followed by a fully connected output layer
    n.ip1 = L.InnerProduct(n.lstm1, num_output=1, bias_term=True, axis=2,
                           weight_filler=dict(type='gaussian',mean=0,std=0.1))
    
    # in this implementation, the loss is calculated over
    # the entire time series, not just the predicted future
    n.fulllabel=L.Concat(n.data,n.label, axis=0)
    n.loss = L.EuclideanLoss(n.ip1,n.fulllabel)
    
    ns=n.to_proto()
    with open(net_path, 'w') as f:
        f.write(str(ns))

gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
```

## Solver definition


```python
#standard Adam solver
def gen_solver():
    sd = caffe.proto.caffe_pb2.SolverParameter()
    sd.random_seed = 0xCAFFE
    sd.train_net = net_path
    sd.test_net.append(net_path)
    sd.test_interval = 500  #
    sd.test_iter.append(100)
    sd.max_iter = n_train
    sd.snapshot = n_train
    sd.display = n_display
    sd.snapshot_prefix = snapshot_prefix
    if USE_GPU:
        sd.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
    else:
        sd.solver_mode = caffe.proto.caffe_pb2.SolverParameter.CPU
    sd.type = 'Adam'
    sd.lr_policy = 'fixed'
    sd.base_lr = n_adamAlpha

    with open(solver_config_path, 'w') as f:
        f.write(str(sd))
gen_solver()

### load the solver and create train and test nets
solver = caffe.get_solver(solver_config_path)
```

## LSTM specific weight initialisation


```python
def init_weights(solver):
    # Initialize forget gate bias to 1
    #
    # from include/caffe/layers/lstm_layer.hpp:
    #/***
    # * The specific architecture used in this implementation is as described in
    # * "Learning to Execute" [arXiv:1410.4615], reproduced below:
    # *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
    # *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
    # *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
    # *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
    # *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
    # *     h_t := o_t .* \tanh[c_t]
    # * In the implementation, the i, f, o, and g computations are performed as a
    # * single inner product.
    # ***/
    #
    # Weights and Biases are stored as a blob of size 4 * n_hidden
    # in order i,f,o,g the caffe LSTM implementation
    # As such [ n_hidden : 2 * n_hidden ] addresses the forget gates
    #
    # This is a common trick to speed up LSTM training.
    # Unfortunately Caffe does not offer selective weight initialisation in the layer definition.
    solver.net.params['lstm1'][1].data[n_hidden : 2 * n_hidden] = 1

init_weights(solver)
```

## Training


```python
# Train network
def train_single(solver, niter, disp_step):
    train_loss = np.zeros(niter) # this is for plotting, later
    
    # the first entry in each time series has its clip mask set to 0
    clipmask = np.ones((n_steps+n_outputs,batch_size))
    clipmask[0,:] = np.zeros((batch_size))
    for i in range(niter):
        _, batch_x, _, batch_y = generate_sample(f=None,
                                         t0=None,
                                         batch_size=batch_size,
                                         samples=n_steps,
                                         predict=n_outputs)
        # IMPORTANT: Caffe LSTM has time in first dimension and batch in second, so
        # batched training data needs to be transposed
        batch_x = batch_x.transpose()
        batch_y = batch_y.transpose()
        solver.net.blobs['label'].data[:,:,0] = batch_y
        solver.net.blobs['data'].data[:,:,0]  = batch_x
        solver.net.blobs['clip'].data[:,:] = clipmask
        solver.step(1)
        train_loss[i] = solver.net.blobs['loss'].data
        if i % disp_step == 0:
            if i==0:
                print "step ", i, ", loss = ", train_loss[i]
            else:
                print "step ", i, ", loss = ", train_loss[i], ", avg loss = ", np.average(train_loss[i-disp_step:i])
    print("Finished training, iteration reached ", niter, " final loss = ", train_loss[niter-1],
        " final avg = ", np.average(train_loss[niter-disp_step:niter-1]))
    return train_loss

train_loss = train_single(solver,n_train,n_display)
#explicitly save snapshot if it has not been done yet
print 'saving snapshot to "%s_iter_%i.caffemodel"' % (snapshot_prefix,n_train)
solver.snapshot()
# plot loss value during training
plt.plot(np.arange(n_train), train_loss)
plt.show()
```

    step  0 , loss =  4.97186183929
    step  200 , loss =  1.89537191391 , avg loss =  2.89020187199
    step  400 , loss =  1.76646876335 , avg loss =  1.79736783981
    step  600 , loss =  1.38966321945 , avg loss =  1.55572113276
    step  800 , loss =  0.879129827023 , avg loss =  1.28479174554
    step  1000 , loss =  0.517951667309 , avg loss =  0.680177893043
    step  1200 , loss =  0.31413936615 , avg loss =  0.361028993651
    step  1400 , loss =  0.14847637713 , avg loss =  0.273928027898
    step  1600 , loss =  0.14750662446 , avg loss =  0.229198574387
    step  1800 , loss =  0.181031450629 , avg loss =  0.166939165331
    step  2000 , loss =  0.089435338974 , avg loss =  0.15380114615
    step  2200 , loss =  0.103752695024 , avg loss =  0.133749665637
    step  2400 , loss =  0.10819543153 , avg loss =  0.118271256164
    step  2600 , loss =  0.0789108201861 , avg loss =  0.125931381173
    step  2800 , loss =  0.0857642516494 , avg loss =  0.105821824614
    step  3000 , loss =  0.045746922493 , avg loss =  0.10222964704
    step  3200 , loss =  0.054507561028 , avg loss =  0.101947768256
    step  3400 , loss =  0.0598913244903 , avg loss =  0.0840673100948
    step  3600 , loss =  0.0560924224555 , avg loss =  0.0918263241276
    step  3800 , loss =  0.0534800291061 , avg loss =  0.0764877726696
    ('Finished training, iteration reached ', 4000, ' final loss = ', 0.097620658576488495, ' final avg = ', 0.077226599631597045)
    saving snapshot to "lstm_demo_snapshot_iter_4000.caffemodel"



![png](output_11_1.png)


## Testing


```python
# Test the prediction with trained (unrolled) net
# we can change the batch size on the network at runtime, but not the number of timesteps (depth of unrolling)
def test_net(net,n_tests):
    batch_size = 1
    net.blobs['data'].reshape(n_steps, batch_size, n_input)
    net.blobs['label'].reshape(n_outputs, batch_size, n_input)
    net.blobs['clip'].reshape(n_steps+n_outputs, batch_size)
    net.blobs['dummy'].reshape(n_outputs, batch_size, n_input)
    net.reshape()

    clipmask = np.ones((n_steps+n_outputs,batch_size))
    clipmask[0,:] = np.zeros(batch_size)

    for i in range(1, n_tests + 1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i+0.1337, t0=None, samples=n_steps, predict=n_outputs)
        test_input = y.transpose()
        expected_y = expected_y.reshape(n_outputs)
        net.blobs['data'].data[:,:,0]  = test_input
        net.blobs['clip'].data[:,:] = clipmask
        net.forward()
        prediction = net.blobs['ip1'].data

        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        t2 = np.append(t,next_t)
        prediction = prediction.squeeze()
        
        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=":")
        plt.plot(t2, prediction, color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')
    plt.show()
test_net(solver.test_nets[0],3)
```


![png](output_13_0.png)


## Network definition for infinite length sequence inference:


```python
#generate inference single time step network
def gen_deploy_network(n_input, n_hidden):
    n = caffe.NetSpec()
    # This network has only scalar input and output data 
    # No unrolling takes place, as such the length of each time series is 1
    # No batch processing either (batch size is 1)
    # The beginning of a new series (discarding LSTM hidden data) is indicated through the clip flag
    n.data, n.clip = L.Input( shape=[ dict(dim=[1,1,n_input]),
                                           dict(dim=[1,1]) ],ntop=2)
    n.lstm1 = L.LSTM(n.data, n.clip,
                     recurrent_param = dict(num_output=n_hidden,
                                            weight_filler = dict(type='uniform',min=-0.08,max=0.08)))
    n.ip1 = L.InnerProduct(n.lstm1, num_output=1, bias_term=True, axis=2,
                           weight_filler=dict(type='gaussian',mean=0,std=0.1))
    # n.ip1 is the output layer
    # there is no loss layer and no loss calculation
    # this network is extremely small and performant, compared to the unrolled training network
    ns=n.to_proto()
    return ns

ns = gen_deploy_network(n_input, n_hidden)
with open(inference_path, 'w') as f:
    f.write(str(ns))

# load Network with weights from previously trained network
# alternatively the network could be loaded without weights and then explicitly assigned (copied) layer by layer
print 'loading snapshot from "%s_iter_%i.caffemodel"' % (snapshot_prefix,n_train)
net = caffe.Net(inference_path,caffe.TEST,weights='%s_iter_%i.caffemodel'%(snapshot_prefix,n_train))
```

    loading snapshot from "lstm_demo_snapshot_iter_4000.caffemodel"


## Inference / Validation


```python
# the single step network can process infinite time series in a loop,
# as such we can increate n_outputs safely to have a glance at long term behaviour

def test_net_iterative(net,n_tests,n_outputs):
    for i in range(1, n_tests + 1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i+0.1337, t0=None, samples=n_steps, predict=n_outputs)
        expected_y = expected_y.reshape(n_outputs)

        net.blobs['clip'].data[0,0]=0
        prediction = []
        for T in range(n_steps):
            net.blobs['data'].data[0,0,0] = y[0,T]
            net.forward()
            prediction.append(net.blobs['ip1'].data[0,0,0])
            net.blobs['clip'].data[0,0]=1
        
        for T in range(n_outputs):
            # in this case we have to manually indicate to the network
            # that there is no more input data at the current time step
            net.blobs['data'].data[0,0,0] = NO_INPUT_DATA
            net.forward()
            prediction.append(net.blobs['ip1'].data[0,0,0])
            net.blobs['clip'].data[0,0]=1
        
        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        t2 = np.append(t,next_t)
        prediction = np.array(prediction)
        
        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=":")
        plt.plot(t2, prediction, color='red')
        plt.ylim([-1, 1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')
    plt.show()

test_net_iterative(net,3,600)
```


![png](output_17_0.png)


### Observation:
The network drifts towards a generic sine wave at constant frequency when left running for longer than the training sample size.
What happens if we train with a longer training window?


```python
n_outputs = 200
n_train = 20000
net_path = 'lstm_demo2_network.train.prototxt'
solver_config_path = 'lstm_demo2_solver.prototxt'
snapshot_prefix = 'lstm_demo2_snapshot'

gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
gen_solver()
solver = caffe.get_solver(solver_config_path)
init_weights(solver)
train_loss = train_single(solver,n_train,800)
plt.plot(np.arange(n_train), train_loss)
plt.show()
```

    step  0 , loss =  4.98372268677
    step  800 , loss =  3.01290464401 , avg loss =  3.44137524188
    step  1600 , loss =  3.4637157917 , avg loss =  2.68596363485
    step  2400 , loss =  1.36267638206 , avg loss =  2.34594154105
    step  3200 , loss =  3.02325534821 , avg loss =  1.89801232122
    step  4000 , loss =  1.2977091074 , avg loss =  1.51855055779
    step  4800 , loss =  0.857801914215 , avg loss =  1.2207886498
    step  5600 , loss =  0.952251911163 , avg loss =  0.965992472395
    step  6400 , loss =  0.789873242378 , avg loss =  0.794555792101
    step  7200 , loss =  0.674143612385 , avg loss =  0.692525207326
    step  8000 , loss =  0.371301561594 , avg loss =  0.651759639904
    step  8800 , loss =  0.255571305752 , avg loss =  0.604751132783
    step  9600 , loss =  0.661503612995 , avg loss =  0.534228966888
    step  10400 , loss =  0.42470061779 , avg loss =  0.49980330782
    step  11200 , loss =  0.290219664574 , avg loss =  0.496096423734
    step  12000 , loss =  0.289145737886 , avg loss =  0.44723227853
    step  12800 , loss =  0.564271271229 , avg loss =  0.428697049562
    step  13600 , loss =  0.185312658548 , avg loss =  0.411151807634
    step  14400 , loss =  0.173889890313 , avg loss =  0.381747617014
    step  15200 , loss =  0.211088508368 , avg loss =  0.369411457228
    step  16000 , loss =  1.42611289024 , avg loss =  0.343723941082
    step  16800 , loss =  0.170716181397 , avg loss =  0.33687596892
    step  17600 , loss =  0.081849090755 , avg loss =  0.293025357043
    step  18400 , loss =  0.291079968214 , avg loss =  0.341382693015
    step  19200 , loss =  0.571784675121 , avg loss =  0.332360383277
    ('Finished training, iteration reached ', 20000, ' final loss = ', 0.20528802275657654, ' final avg = ', 0.38653186177170473)



![png](output_19_1.png)



```python
net.copy_from('%s_iter_%i.caffemodel'%(snapshot_prefix,n_train))
test_net_iterative(net,3,600)
```


![png](output_20_0.png)


### Observation:
With the longer unrolling window, the training converges much slower. The loss accumulated at the end of the window needs to backpropagate many steps until it reaches a timestep in which there is still a useful memory in the LSTM layer. This makes training potentially unstable.

A better option is to attempt iterative fine tuning with slowly increasing time windows. As a bonus, this allows doing most of the training with shorter windows, which means smaller networks and faster computation.

## Iterative Finetuning


```python
n_display = 400
n_outputs = 50
n_train = 4500
net_path = 'lstm_demo3_0_network.train.prototxt'
solver_config_path = 'lstm_demo3_0_solver.prototxt'
snapshot_prefix = 'lstm_demo3_0_snapshot'

print "initial training ", n_outputs," ouput timesteps for ",n_train," training cycles"
gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
gen_solver()
solver = caffe.get_solver(solver_config_path)
init_weights(solver)
train_loss = train_single(solver,n_train,n_display)
plt.plot(np.arange(n_train), train_loss)
plt.show()
net_0 = '%s_iter_%i.caffemodel'%(snapshot_prefix,n_train)
net.copy_from(net_0)
test_net_iterative(net,3,600)
```

    initial training  50  ouput timesteps for  4500  training cycles
    step  0 , loss =  4.99590826035
    step  400 , loss =  1.66586947441 , avg loss =  2.27522636026
    step  800 , loss =  1.01261091232 , avg loss =  1.30000257283
    step  1200 , loss =  0.249216228724 , avg loss =  0.4566435403
    step  1600 , loss =  0.226823598146 , avg loss =  0.23752257539
    step  2000 , loss =  0.122280284762 , avg loss =  0.179536702372
    step  2400 , loss =  0.114655710757 , avg loss =  0.142414935324
    step  2800 , loss =  0.109308563173 , avg loss =  0.115479582045
    step  3200 , loss =  0.0844941660762 , avg loss =  0.109894855414
    step  3600 , loss =  0.0452548526227 , avg loss =  0.0877064392809
    step  4000 , loss =  0.0511142835021 , avg loss =  0.0730762454495
    step  4400 , loss =  0.0494143515825 , avg loss =  0.068965257192
    ('Finished training, iteration reached ', 4500, ' final loss = ', 0.044924627989530563, ' final avg = ', 0.06679526652515233)



![png](output_23_1.png)



![png](output_23_2.png)



```python
n_outputs = 100
n_train = 6000
net_path = 'lstm_demo3_1_network.train.prototxt'
solver_config_path = 'lstm_demo3_1_solver.prototxt'
snapshot_prefix = 'lstm_demo3_1_snapshot'

print "finetuning ", n_outputs," ouput timesteps for ",n_train," training cycles"
gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
gen_solver()
solver = caffe.get_solver(solver_config_path)
solver.net.copy_from(net_0)
train_loss = train_single(solver,n_train,n_display)
plt.plot(np.arange(n_train), train_loss)
plt.show()
net_1 = '%s_iter_%i.caffemodel'%(snapshot_prefix,n_train)
net.copy_from(net_1)
test_net_iterative(net,3,600)
```

    finetuning  100  ouput timesteps for  6000  training cycles
    step  0 , loss =  0.917309701443
    step  400 , loss =  0.21685898304 , avg loss =  0.383107975218
    step  800 , loss =  0.129005819559 , avg loss =  0.268622112088
    step  1200 , loss =  0.527812898159 , avg loss =  0.242106551025
    step  1600 , loss =  0.313889920712 , avg loss =  0.205841700919
    step  2000 , loss =  0.112634092569 , avg loss =  0.182493864708
    step  2400 , loss =  0.0576966963708 , avg loss =  0.172911731806
    step  2800 , loss =  0.115338936448 , avg loss =  0.149683445292
    step  3200 , loss =  0.0543339364231 , avg loss =  0.158921282981
    step  3600 , loss =  0.0945870131254 , avg loss =  0.124688625946
    step  4000 , loss =  0.180501759052 , avg loss =  0.126365286503
    step  4400 , loss =  0.126055464149 , avg loss =  0.118555996083
    step  4800 , loss =  0.113133296371 , avg loss =  0.115533719715
    step  5200 , loss =  0.235036760569 , avg loss =  0.121917440048
    step  5600 , loss =  0.053582854569 , avg loss =  0.1189321648
    ('Finished training, iteration reached ', 6000, ' final loss = ', 0.35921072959899902, ' final avg = ', 0.093345553023661293)



![png](output_24_1.png)



![png](output_24_2.png)



```python
n_outputs = 200
n_train = 8000
net_path = 'lstm_demo3_2_network.train.prototxt'
solver_config_path = 'lstm_demo3_2_solver.prototxt'
snapshot_prefix = 'lstm_demo3_2_snapshot'

print "finetuning ", n_outputs," ouput timesteps for ",n_train," training cycles"
gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
gen_solver()
solver = caffe.get_solver(solver_config_path)
solver.net.copy_from(net_1)
train_loss = train_single(solver,n_train,n_display)
plt.plot(np.arange(n_train), train_loss)
plt.show()
net_2 = '%s_iter_%i.caffemodel'%(snapshot_prefix,n_train)
net.copy_from(net_2)
test_net_iterative(net,3,600)
```

    finetuning  200  ouput timesteps for  8000  training cycles
    step  0 , loss =  2.78915858269
    step  400 , loss =  0.756085991859 , avg loss =  0.570729123149
    step  800 , loss =  0.167720377445 , avg loss =  0.516727606393
    step  1200 , loss =  0.136021032929 , avg loss =  0.441396284606
    step  1600 , loss =  0.199986502528 , avg loss =  0.395003471971
    step  2000 , loss =  0.142788201571 , avg loss =  0.373432880677
    step  2400 , loss =  0.0898929983377 , avg loss =  0.361235053688
    step  2800 , loss =  0.276249110699 , avg loss =  0.357156292014
    step  3200 , loss =  0.36933863163 , avg loss =  0.311579896398
    step  3600 , loss =  0.355469763279 , avg loss =  0.328943576626
    step  4000 , loss =  0.129715204239 , avg loss =  0.312640832141
    step  4400 , loss =  0.175822347403 , avg loss =  0.309760584794
    step  4800 , loss =  0.0813407376409 , avg loss =  0.315446726382
    step  5200 , loss =  0.879916787148 , avg loss =  0.228818979207
    step  5600 , loss =  0.34600096941 , avg loss =  0.279229280008
    step  6000 , loss =  0.130669862032 , avg loss =  0.255788187869
    step  6400 , loss =  0.511739313602 , avg loss =  0.25297228381
    step  6800 , loss =  0.0757004618645 , avg loss =  0.264342094548
    step  7200 , loss =  0.0969089642167 , avg loss =  0.227965111686
    step  7600 , loss =  0.479567170143 , avg loss =  0.24966308685
    ('Finished training, iteration reached ', 8000, ' final loss = ', 0.10637857764959335, ' final avg = ', 0.22641879402306445)



![png](output_25_1.png)



![png](output_25_2.png)



```python
n_outputs = 400
n_train = 10000
net_path = 'lstm_demo3_3_network.train.prototxt'
solver_config_path = 'lstm_demo3_3_solver.prototxt'
snapshot_prefix = 'lstm_demo3_3_snapshot'

print "finetuning ", n_outputs," ouput timesteps for ",n_train," training cycles"
gen_network(n_steps, n_outputs, batch_size, n_input, n_hidden)
gen_solver()
solver = caffe.get_solver(solver_config_path)
solver.net.copy_from(net_2)
train_loss = train_single(solver,n_train,n_display)
plt.plot(np.arange(n_train), train_loss)
plt.show()
net_3 = '%s_iter_%i.caffemodel'%(snapshot_prefix,n_train)
net.copy_from(net_3)
test_net_iterative(net,3,600)
```

    finetuning  400  ouput timesteps for  10000  training cycles
    step  0 , loss =  1.37016475201
    step  400 , loss =  0.661862552166 , avg loss =  1.12462965481
    step  800 , loss =  0.71813762188 , avg loss =  1.01753102593
    step  1200 , loss =  0.706093072891 , avg loss =  0.925081891268
    step  1600 , loss =  0.879032492638 , avg loss =  0.936979228035
    step  2000 , loss =  0.842401206493 , avg loss =  0.900434936769
    step  2400 , loss =  0.468837291002 , avg loss =  0.816657312065
    step  2800 , loss =  0.838852465153 , avg loss =  0.815633930974
    step  3200 , loss =  0.333592921495 , avg loss =  0.780244623274
    step  3600 , loss =  0.699046254158 , avg loss =  0.759092914686
    step  4000 , loss =  1.22850072384 , avg loss =  0.691919682287
    step  4400 , loss =  0.503066062927 , avg loss =  0.749729227796
    step  4800 , loss =  0.267374068499 , avg loss =  0.714126435891
    step  5200 , loss =  0.731412708759 , avg loss =  0.609350840729
    step  5600 , loss =  0.308685421944 , avg loss =  0.664788552076
    step  6000 , loss =  0.262960672379 , avg loss =  0.631102285534
    step  6400 , loss =  0.143768370152 , avg loss =  0.583463267498
    step  6800 , loss =  0.141827791929 , avg loss =  0.543628830761
    step  7200 , loss =  0.311824768782 , avg loss =  0.626280408092
    step  7600 , loss =  1.02792584896 , avg loss =  0.577311622147
    step  8000 , loss =  0.173835977912 , avg loss =  0.575452747773
    step  8400 , loss =  0.225577503443 , avg loss =  0.52591862252
    step  8800 , loss =  1.35598087311 , avg loss =  0.573785225395
    step  9200 , loss =  0.787913620472 , avg loss =  0.489780008737
    step  9600 , loss =  0.295411169529 , avg loss =  0.515823517349
    ('Finished training, iteration reached ', 10000, ' final loss = ', 0.11684667319059372, ' final avg = ', 0.47138720620096775)



![png](output_26_1.png)



![png](output_26_2.png)


### Long term test


```python
test_net_iterative(net,3,1500)
```


![png](output_28_0.png)


Trained with sufficient long unrolled time window, the resulting network is capable of identifying frequency and phase of the sin() wave with high accuracy and generate a time-stable reproduction.
