""" test_sf.py
The goal of this file is to analyze the network performances for different sizes
of entry signals.

For each size, we have 5 different signals that we want to be able to recognize.
Ex:
| size |   sig 1    |   sig 2    |   sig 3    |   sig 4    |   sig 5    |     
|   5  |      10000 |      01000 |      00100 |      00010 |      00001 |
|  10  | 1100000000 | 0011000000 | 0000110000 | 0000001100 | 0000000011 |
"""

import brian2 as b2
import numpy as np
import sys
import time

def create_training_network(n_input, n_e, n_i):
    """Create a network suitable for training with stdp mechanism.

    Params:
    n_input - the number of input neurons
    n_e - the number of excitatory neurons
    n_i - the number of inhibitatory neurons

    Returns the network
    """
    neuron_eqs_e = '''
        dv/dt = (v_rest_e - gi * 85 * mV - v * (1 + gi + ge)) / (10 * ms) : volt
        dgi/dt = -gi / (2 * ms) : 1
        dge/dt = -ge / (1 * ms) : 1
        dtheta/dt = -theta / (1e5 * ms) : volt
        '''
    
    reset_e = 'v = v_reset_e; theta += 0.05 * mV'

    neuron_eqs_i = '''
        dv/dt = (v_rest_i - v * (1 + ge)) / (10 * ms) : volt
        dge/dt = -ge / (1 * ms) : 1
        '''
    
    neuron_group_e = b2.NeuronGroup(n_e, neuron_eqs_e, threshold='v > theta + v_thresh_e', reset=reset_e, refractory=refrac_e, method='euler', name="n_e")
    neuron_group_i = b2.NeuronGroup(n_i, neuron_eqs_i, threshold='v > v_thresh_i', reset='v = v_reset_i', refractory=refrac_i, method='euler', name="n_i")

    neuron_group_e.v = v_rest_e - 40 * b2.mV
    neuron_group_i.v = v_rest_i - 40 * b2.mV

    synapses_ei = b2.Synapses(neuron_group_e, neuron_group_i, 'w : 1', on_pre='ge_post += w', name="s_ei")
    synapses_ei.connect(j='i')
    synapses_ei.w = 15 #  10.4

    synapses_ie = b2.Synapses(neuron_group_i, neuron_group_e, 'w : 1', on_pre='gi_post += w', name="s_ie")
    synapses_ie.connect(condition='j != i')
    synapses_ie.w = 17

    input_group_pos = b2.PoissonGroup(n_input, 0 * b2.Hz, name="n_inp")
    
    input_group_neg = b2.PoissonGroup(n_input, 0 * b2.Hz, name="n_inn")
    
    

    synapse_eqs_stdp_pos = '''
                       w : 1
                       post2before: 1
                       dapre/dt = -apre / tau_pre : 1 (clock-driven)
                       dapost/dt = -(apost-0.3) / tau_post : 1 (clock-driven)
                       dapost2/dt = -apost2 / tau_post2 : 1 (clock-driven)
                       '''
    synapse_eqs_pre_pos = '''
                      ge_post += w
                      apre = 1
                      w = clip(w - 0.0001 * apost, 0, 1)
                      w_neg = synapses_input_neg.w[postsynaptic_indices]
                      synapses_input_neg.w[postsynaptic_indices] = clip(w_neg - 0.01 * apre * w, 0, 1)
                      '''
    synapse_eqs_post_pos = '''
                       post2before = apost2
                       w = clip(w + 0.01 * apre * post2before, 0, 1)
                       apost = 1
                       apost2 = 1
                       '''
    
    synapse_eqs_stdp_neg = '''
                       w : 1
                       post2before: 1
                       dapre/dt = -apre / tau_pre : 1 (clock-driven)
                       dapost/dt = -(apost-0.3) / tau_post : 1 (clock-driven)
                       dapost2/dt = -apost2 / tau_post2 : 1 (clock-driven)
                       '''
    synapse_eqs_pre_neg = '''
                      ge_post += w
                      apre = 1
                      w = clip(w - 0.0001 * apost, 0, 1)
                      w_pos = synapses_input_pos.w[postsynaptic_indices]
                      synapses_input_pos.w[postsynaptic_indices] = clip(w_pos - 0.01 * apre * w, 0, 1)
                      '''
    synapse_eqs_post_neg = '''
                       post2before = apost2
                       w = clip(w + 0.01 * apre * post2before, 0, 1)
                       apost = 1
                       apost2 = 1
                       '''
    
    synapses_input_pos = b2.Synapses(input_group_pos, neuron_group_e, synapse_eqs_stdp_pos, on_pre=synapse_eqs_pre_pos, 
                                     on_post=synapse_eqs_post_pos.format(postsynaptic_indices=postsynaptic_indices), method='exact', name="s_inpe")

    synapses_input_pos.connect()
    synapses_input_pos.w = [b2.random() * 0.2 + 0.2 for i in range(n_input*n_e)]
    
    synapses_input_neg = b2.Synapses(input_group_neg, neuron_group_e, synapse_eqs_stdp_neg, on_pre=synapse_eqs_pre_neg, 
                                     on_post=synapse_eqs_post_neg.format(postsynaptic_indices=postsynaptic_indices), method='exact', name="s_inne")

    synapses_input_neg.connect()
    synapses_input_neg.w = [b2.random() * 0.2 + 0.2 for i in range(n_input*n_e)]
    
    postsynaptic_indices_pos = synapses_input_pos.j
    postsynaptic_indices_neg = synapses_input_neg.j
    postsynaptic_indices = np.intersect1d(postsynaptic_indices_pos, postsynaptic_indices_neg)

    groups = [input_group_pos, input_group_neg, neuron_group_e, neuron_group_i, synapses_ei, synapses_ie, synapses_input_pos, synapses_input_neg]
    # monitors = [b2.StateMonitor(neuron_group_e, ['v', 'gi', 'theta'], record=True, name="M"), b2.StateMonitor(neuron_group_i, ['v', 'ge'], record=True, name="M4"), b2.StateMonitor(synapses_input, ['w', 'apre', 'apost', 'apost2'], record=True, name="M3")]
    net = b2.Network(groups)  # , monitors)

    return net


def create_testing_network(n_input, n_e, n_i, suffix="0"):
    """Create a network suitable for testing, i.e. without stdp mechanism.

    Params:
    n_input - the number of input neurons
    n_e - the number of excitatory neurons
    n_i - the number of inhibitatory neurons
    suffix - a string allowing to load data previously saved on hard disk

    Returns the network
    """
    neuron_eqs_e = '''
        dv/dt = (v_rest_e - gi * 85 * mV - v * (1 + gi + ge)) / (10 * ms) : volt
        dgi/dt = -gi / (2 * ms) : 1
        dge/dt = -ge / (1 * ms) : 1
        theta : volt
        '''
    
    reset_e = 'v = v_reset_e'

    neuron_eqs_i = '''
        dv/dt = (v_rest_i - v * (1 + ge)) / (10 * ms) : volt
        dge/dt = -ge / (1 * ms) : 1
        '''
    
    neuron_group_e = b2.NeuronGroup(n_e, neuron_eqs_e, threshold='v > theta + v_thresh_e', reset=reset_e, refractory=refrac_e, method='euler', name="n_e")
    neuron_group_i = b2.NeuronGroup(n_i, neuron_eqs_i, threshold='v > v_thresh_i', reset='v = v_reset_i', refractory=refrac_i, method='euler', name="n_i")

    neuron_group_e.v = v_rest_e - 40 * b2.mV
    neuron_group_i.v = v_rest_i - 40 * b2.mV

    if suffix != "":
        neuron_group_e.theta = np.load('storage/theta/e_' + suffix + '.npy') * b2.volt

    synapses_ei = b2.Synapses(neuron_group_e, neuron_group_i, 'w : 1', on_pre='ge_post += w', name="s_ei")
    synapses_ei.connect(j='i')
    synapses_ei.w = 15 #  10.4

    synapses_ie = b2.Synapses(neuron_group_i, neuron_group_e, 'w : 1', on_pre='gi_post += w', name="s_ie")
    synapses_ie.connect(condition='j != i')
    synapses_ie.w = 17

    input_group_pos = b2.PoissonGroup(n_input, 0 * b2.Hz, name="n_inp")
    
    synapses_input_pos = b2.Synapses(input_group_pos, neuron_group_e, 'w : 1', on_pre='ge_post += w', method='exact', name="s_inpe")
    
    input_group_neg = b2.PoissonGroup(n_input, 0 * b2.Hz, name="n_inn")
    
    synapses_input_neg = b2.Synapses(input_group_neg, neuron_group_e, 'w : 1', on_pre='ge_post += w', method='exact', name="s_inne")

    synapses_input_pos.connect()
    if suffix != "":
        synapses_input_pos.w = np.load('storage/weights/input_' + suffix + '.npy')
        
    synapses_input_neg.connect()
    if suffix != "":
        synapses_input_neg.w = np.load('storage/weights/input_' + suffix + '.npy')

    groups = [input_group_pos, input_group_neg, neuron_group_e, neuron_group_i, synapses_ei, synapses_ie, synapses_input_pos, synapses_input_neg]
    # monitors = [b2.StateMonitor(neuron_group_e, ['v', 'gi', 'theta'], record=True, name="M"), b2.StateMonitor(neuron_group_i, ['v', 'ge'], record=True, name="M4"), b2.StateMonitor(synapses_input, ['w', 'apre', 'apost', 'apost2'], record=True, name="M3")]
    net = b2.Network(groups)

    return net


if __name__ == "__main__":
    train_mode = True
    if sys.argv.__contains__("--train"):
        train_mode = True

    """ We want a 20-entries network, able to recognize 5 different shapes

    11110000000000000000 ; 00001111000000000000 ; 00000000111100000000 ; 00000000000011110000 ; 00000000000000001111
    """

    # Network parameters
    n_input = 20
    n_e = 5
    n_i = n_e
    single_example_time = 0.35 * b2.second
    unique_spike_time = 0.005 * b2.second
    resting_time = 0.15 * b2.second

    v_rest_e = -65 * b2.mV
    v_rest_i = -60 * b2.mV
    v_reset_e = -65 * b2.mV
    v_reset_i = -45 * b2.mV
    v_thresh_e = -60 * b2.mV
    v_thresh_i = -40 * b2.mV
    refrac_e = 5 * b2.ms
    refrac_i = 2 * b2.ms

    tau_pre = tau_post = 20 * b2.ms
    tau_post2 = 40 * b2.ms
    A_pre = 0.01
    A_post = -A_pre * 1.02  # * tau_pre / tau_post * 1.05

    # signals = np.asarray([
    #     [1, 1, 1, 1,    0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 0], 
    #     [0, 0, 0, 0,    1, 1, 1, 1,     0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 0], 
    #     [0, 0, 0, 0,    0, 0, 0, 0,     1, 1, 1, 1,     0, 0, 0, 0,     0, 0, 0, 0], 
    #     [0, 0, 0, 0,    0, 0, 0, 0,     0, 0, 0, 0,     1, 1, 1, 1,     0, 0, 0, 0], 
    #     [0, 0, 0, 0,    0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 0,     1, 1, 1, 1], 
    # ])

    def noise(sig, h, sdev):
        """Generate noise on the digits of a given signal
        """
        sig = [digit * h + np.random.normal(0, sdev) for digit in sig]
        # sig = [min(max(digit, 0), 1) for digit in sig]
        sig = [max(digit, 0) for digit in sig]
        return np.asarray(sig)
    
    def save_choices(choices, i, j):
        """Save the choices made by the network for further plotting
        """
        # print("L, K =", i, j)
        with open("storage/choices/choices_" + str(i) + "_" + str(j) + ".txt", "a") as f:
            for c in choices:
                f.write(str(c))

    def fill_weights(sf, neg):
        """Generate the ideal weights so that the network is able to recognize
        the 5 different signals. It's basically the transpose of the signals matrix
        """
        if neg :
            return [0 if i // sf == i % n_e else 1 for i in range(sf * n_e)]
        else :
            return [1 if i // sf == i % n_e else 0 for i in range(sf * n_e)]

    def gen_signals(sf):
        """Generate 5 (orthogonal) signals with given length

        sf must be a multiple of 5 for the signals to be orthogonal
        """
        return np.asarray([[-1 if i // (sf / n_e) == j else 1 for i in range(sf)] for j in range(n_e)])
    
    def convert_neg_signal(sig) :
        """
        Return 2 signals for negative and positive entries of the network 
        """
        list1 = np.asarray([[0 for i in range(len(sig[0]))] for j in range(len(sig))])
        list2 = np.asarray([[0 for i in range(len(sig[0]))] for j in range(len(sig))])
        
        for j in range(len(sig)) :
            for i in range(len(sig[j])) :
                if sig[j][i] > 0 :
                    list1[j][i] = sig[j][i]
                else :
                    list2[j][i] = -sig[j][i]
        
        return(list1, list2)
                

    # Simulation
    sim_start_time = time.time()
    perf = []
    print("Running simulation...")
    for l in range(11):
        print("Computing curve nb "+str(l))
        perf.append(([], []))
        sf = 5
        for k in range(21):
            print("[" + str(k) + "/20] > Testing for sf=" + str(sf))
            if train_mode:
                training_network = create_training_network(sf, n_e, n_i)
                input_group_pos = training_network['n_inp']
                input_group_neg = training_network['n_inn']
                neuron_group_e = training_network['n_e']
                synapses_input_pos = training_network['s_inpe']
                synapses_input_neg = training_network['s_inne']

            testing_network = create_testing_network(sf, n_e, n_i, suffix="")
            signals = gen_signals(sf)
            sig1, sig2 = convert_neg_signal(signals)

            if train_mode:
                neuron_group_e.theta = 0 * b2.volt
                synapses_input_pos.w = [b2.random() * 0.2 + 0.2 for i in range(sf * n_e)]
                synapses_input_neg.w = [b2.random() * 0.2 + 0.2 for i in range(sf * n_e)]
                #synapses_input_pos.w = fill_weights(sf, False)
                #synapses_input_neg.w = fill_weights(sf, True)

            if train_mode:
                # This is where we train the network (stdp mechanism changes the weights)
                print(" > Training... 000/100", end="", flush=True)
                for i in range(100):
                    print(f"\033[s\033[" + str(4 + len(str(i+1))) + "D" + str(i+1) + "\033[u", end="", flush=True) # just a hack to update the quota on a single line
                    input_group_pos.rates = sig1[i % 5] * 63.75 * b2.Hz
                    input_group_neg.rates = sig2[i % 5] * 63.75 * b2.Hz
                    training_network.run(single_example_time)
                    input_group_pos.rates = 0
                    input_group_neg.rates = 0
                    training_network.run(resting_time)

            if train_mode:
                print("\n > Saving theta and weights...")
                np.save("storage/weights/input_" + str(l) + "_" + str(k), np.copy(synapses_input_pos.w))
                np.save("storage/theta/e_" + str(l) + "_" + str(k), neuron_group_e.theta)

            #
            # Classification
            #
            # testing_network['n_e'].theta = np.load("storage/theta/e_"+str(l)+".npy") * b2.volt
            # testing_network['s_ine'].w = np.load("storage/weights/input_"+str(l)+".npy")
            testing_network['n_e'].theta = neuron_group_e.theta
            testing_network['s_inpe'].w = synapses_input_pos.w
            testing_network['s_inne'].w = synapses_input_neg.w

            # This is where we assign a neuron to a signal, because initially we have no idea
            # which neuron has been trained to detect which signal
            print(" > Classifying... 00/50", end="", flush=True)
            spike_counter = b2.SpikeMonitor(testing_network["n_e"], record=False, name="counter")
            testing_network.add(spike_counter)
            spike_count = [np.asarray([0] * 5), np.asarray([0] * 5), np.asarray([0] * 5), np.asarray([0] * 5), np.asarray([0] * 5)]
            last_spike_count = np.asarray([0] * 5)
            classes = [-1] * 5

            # We present a bunch of signals and save the number of spikes for each neuron
            for i in range(50):
                print(f"\033[s\033[" + str(3 + len(str(i+1))) + "D" + str(i+1) + "\033[u", end="", flush=True)
                
                testing_network["n_inp"].rates = sig1[i % 5] * 63.75 * b2.Hz
                testing_network["n_inn"].rates = sig2[i % 5] * 63.75 * b2.Hz
                testing_network.run(single_example_time)
                testing_network["n_inp"].rates = 0
                testing_network["n_inn"].rates = 0
                testing_network.run(resting_time)

                for j in range(5):
                    spike_count[j][i % 5] += spike_counter.count[j] - last_spike_count[j]
                last_spike_count = np.copy(spike_counter.count)
            
            # We assign to each neuron the signal he has spiked to the most
            # Ex: if we stored the following spikes for neuron 1: [12, 2, 42, 15, 17], we assign it the signal 2 (start count from 0)
            # TODO: 2 neurons can have the same class (very rare thanks to homeostasis), but shouldn't
            for j in range(5):
                classes[j] = np.argmax(spike_count[j])
            print(" => " + str(classes))

            testing_network.remove(spike_counter)

            test_start_time = time.time()
            spike_counter = b2.SpikeMonitor(testing_network["n_e"], record=False, name="counter")
            testing_network.add(spike_counter)
            last_spike_count = np.asarray([0] * 5)

            #
            # Calcul des performances
            #
            # This is where we test performances
            # We present a lot of signals and check if the network can recognize them correctly
            print(" > Calculating performances... 0000/1000", end="", flush=True)
            choices = []
            for i in range(1001):
                print(f"\033[s\033[" + str(6 + len(str(i+1))) + "D" + str(i+1) + "\033[u", end="", flush=True)                
                
                testing_network["n_inp"].rates = sig1[i % 5] * 63.75 * b2.Hz
                testing_network["n_inn"].rates = sig2[i % 5] * 63.75 * b2.Hz
                testing_network.run(single_example_time)
                testing_network["n_inp"].rates = 0
                testing_network["n_inn"].rates = 0
                testing_network.run(resting_time)

                spike_count = np.asarray(spike_counter.count) - last_spike_count
                last_spike_count = np.copy(spike_counter.count)
                choices.append(classes[np.argmax(spike_count)])

                # Choices are stored over the course of the simulation (checkpoints)
                if i % 1000 == 0 and i != 0:
                    save_choices(choices[i-1000:i], l, k)

            testing_network.remove(spike_counter)

            # We can compute the success rate during simulation,
            # but it's just cosmetics because we'll use the stored data to make a graph
            success = 0
            for i in range(len(choices)):
                if choices[i] == i % 5:
                    success += 1
            print("\nSuccess rate: ", success / len(choices))
            perf[l][0].append(sf)
            perf[l][1].append(success / len(choices))

            # We change the size of signals for the next point
            sf += 5

            print("Time elapsed", "{:.1f}".format(time.time() - test_start_time), "s")

    print("Time elapsed since start of simulation: ", "{:.1f}".format(time.time() - sim_start_time), "s")

    # b2.figure(1)
    # b2.plot(M.t/b2.ms, M.v[0]/b2.mV, label='Neuron 0 e')
    # b2.plot(M.t/b2.ms, M.v[1]/b2.mV, label='Neuron 1 e')
    # b2.plot(M.t/b2.ms, M.v[2]/b2.mV, label='Neuron 2 e')
    # b2.plot(M4.t/b2.ms, M4.v[0]/b2.mV, label='Neuron 0 i')
    # b2.plot(M4.t/b2.ms, M4.v[1]/b2.mV, label='Neuron 1 i')
    # b2.plot(M4.t/b2.ms, M4.v[2]/b2.mV, label='Neuron 2 i')
    # b2.xlabel('Time (ms)')
    # b2.ylabel('v (mV)')
    # b2.legend()

    # b2.figure(4)
    # b2.subplot(211)
    # b2.plot(M.t/b2.ms, M.gi[1], label='=Neuron 1 e gi')
    # b2.plot(M4.t/b2.ms, M4.ge[0], label='=Neuron 0 i ge')
    # b2.ylabel('gi / ge (nS)')
    # b2.legend()

    # b2.figure(2)
    # b2.subplot(411)
    # b2.plot(M3.t[::10]/b2.ms, M3.apre[0][::10], label='apre')
    # b2.plot(M3.t[::10]/b2.ms, M3.apost[0][::10], label='apost')
    # b2.plot(M3.t[::10]/b2.ms, M3.apost2[0][::10], label='apost2')
    # b2.ylabel('Synapyse 0 -> 0')
    # b2.legend()
    # b2.subplot(411)
    # b2.plot(M.t[::10]/b2.ms, M.theta[0][::10]/b2.mV, label='theta0')
    # b2.plot(M.t[::10]/b2.ms, M.theta[1][::10]/b2.mV, label='theta1')
    # b2.plot(M.t[::10]/b2.ms, M.theta[2][::10]/b2.mV, label='theta2')
    # b2.ylabel('Theta (mV)')
    # b2.legend()
    # b2.subplot(412)
    # b2.plot(M3.t[::10]/b2.ms, M3.w[0][::10], label='w0')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[5][::10], label='w1')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[10][::10], label='w2')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[15][::10], label='w3')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[20][::10], label='w4')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[25][::10], label='w5')
    # b2.ylabel('Weights i,0')
    # b2.legend()
    # b2.subplot(413)
    # b2.plot(M3.t[::10]/b2.ms, M3.w[16][::10], label='w0')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[21][::10], label='w1')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[26][::10], label='w2')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[31][::10], label='w3')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[36][::10], label='w4')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[41][::10], label='w5')
    # b2.ylabel('Weights i,1')
    # b2.legend()
    # b2.subplot(414)
    # b2.plot(M3.t[::10]/b2.ms, M3.w[37][::10], label='w0')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[42][::10], label='w1')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[47][::10], label='w2')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[52][::10], label='w3')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[57][::10], label='w4')
    # b2.plot(M3.t[::10]/b2.ms, M3.w[62][::10], label='w5')
    # b2.ylabel('Weights i,2')
    # b2.legend()
    # b2.xlabel("Time (ms)")

    # b2.figure(3)
    # b2.plot(M2.t/b2.ms, M2.i, '.')
    # b2.xlabel('Time (ms)')
    # b2.ylabel('Rate (Hz)')
    # b2.legend()

    # b2.figure(4)
    # for l in range(3):
    #     b2.plot(perf[l][0], perf[l][1])
    # b2.xlabel('Standard deviation')
    # b2.ylabel('Accuracy')
    # b2.show()