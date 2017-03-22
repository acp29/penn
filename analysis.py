## Penn lab python module for Stimfit
## version 18 November 2016

# load required modules
try:
    import stf
except:
    print "stf could not be imported"
import numpy as np
try:
    import spells
except:
    print "spells could not be imported"
try:
    from scipy import optimize
except:
    print "Optimize module from Scipy could not be imported"
try:
    from scipy import interpolate
except:
    print "Interpolate module from Scipy could not be imported"

def loadmat():
    """
    Load electrophysiology recordings from ephysIO
    HDF5-based Matlab v7.3 (.mat) files 
    """
    
    # Import required modules for file IO
    from Tkinter import Tk
    import tkFileDialog
    from gc import collect

    # Use file open dialog to obtain file path
    root = Tk()
    opt = dict(defaultextension='.mat',filetypes=[('MATLAB v7.3 (HDF5) file','*.mat'), ('All files','*.*')])
    if 'loadcwd' not in globals():
        global loadcwd
    else:
        opt['initialdir'] = loadcwd
    filepath = tkFileDialog.askopenfilename(**opt)
    root.withdraw()
    
    if filepath != '':

        # Move to file directory and check file version
        loadcwd = filepath.rsplit('/',1)[0]
        from os import chdir
        print filepath
        chdir(loadcwd)

        # Load data into python
        import ephysIO
        data = ephysIO.MATload(filepath)

        # Display data in Stimfit
        import stf
        if data.get('xdiff') > 0:
            if data.get('yunit') == "V":
                stf.new_window_list(1.0e+3 * np.array(data.get('array')[1::]))
                stf.set_yunits('m'+data.get('yunit'))
            elif data.get('yunit') == "A":
                stf.new_window_list(1.0e+12 * data.get('array')[1::])
                stf.set_yunits('p'+data.get('yunit'))
            else:
                stf.new_window_list(data.get('array')[1::])
                stf.set_yunits(data.get('yunit'))  
            stf.set_sampling_interval(1.0e+3 * data.get('xdiff'))
            stf.set_xunits('m'+data.get('xunit'))
            stf.set_trace(0)
            stf.set_recording_comment('\n'.join(data['notes']))
            if data['saved']!='':
                date = data['saved'][0:8]
                date = tuple(map(int,(date[0:4],date[4:6],date[6:8])))
                stf.set_recording_date('%s-%s-%s'%date)
                time = data['saved'][9::]
                time = tuple(map(int,(time[0:2],time[2:4],time[4:6])))
                stf.set_recording_time('%i-%i-%i'%time)
        elif data.get('xdiff') == 0:
            raise ValueError("Sample interval is not constant")
        
    else:
        
        data = {}
        
    collect()
        
    return

def savemat():
    """
    Save electrophysiology recordings to ephysIO HDF5-based Matlab
    v7.3 (.mat) files 
    """

    # Import required modules for file IO
    from Tkinter import Tk
    import tkFileDialog
    from gc import collect

    # Use file save dialog to obtain file path
    root = Tk()
    opt = dict(defaultextension='.mat',filetypes=[('MATLAB v7.3 (HDF5) file','*.mat'), ('All files','*.*')])
    if 'savecwd' not in globals():
        global savecwd
    else:
        opt['initialdir'] = savecwd
    filepath = tkFileDialog.asksaveasfilename(**opt)
    root.destroy()
    
    if filepath != '':

        # Move to file directoty
        savecwd = filepath.rsplit('/',1)[0]
        import os
        print filepath
        os.chdir(savecwd)
        filename = filepath.rsplit('/',1)[1]

        # Get data from active Stimfit window
        import stf
        import numpy as np
        n = stf.get_size_channel()
        array = np.array([stf.get_trace(i).tolist() for i in range(n)])
        if np.any(np.isnan(array)) | np.any(np.isinf(array)):
            raise ValueError("nan and inf values cannot be parsed into ephysIO")
        if stf.get_yunits() == 'pA':
            yunit = 'A'
            array = 1.0e-12 * array
        elif stf.get_yunits() == 'mV':
            yunit = 'V'
            array = 1.0e-03 * array
        else:
            yunit = stf.get_yunits()
            #print "Warning: Expected Y dimension units to be either pA or mV"

        # Create X dimension properties
        if stf.get_xunits() == 'ms':
            xunit = 's'
            xdiff = np.array([[1.0e-03 * stf.get_sampling_interval()]])
        else:
            raise ValueError("Expected X dimension units to be ms")
        
        # Calculate X dimension and add to array
        x = xdiff * np.arange(0.0,np.shape(array)[1],1,'float64')
        array = np.concatenate((np.array(x,ndmin=2),array),0)

        # Get data recording notes
        notes = stf.get_recording_comment().split('\n')
        names = None
        
        import ephysIO
        ephysIO.MATsave(filepath, array, xunit, yunit, names, notes)

    collect()
    
    return 

def loadacq4(channel = 1):
    """
    Load electrophysiology recording data from acq4 hdf5 (.ma) files.
    By default the primary recording channel is loaded.  
    
    If the file is in a folder entitled 000, loadacq4 will load
    the recording traces from all sibling folders (000,001,002,...)
    """
    
    # Import required modules for file IO
    from Tkinter import Tk
    import tkFileDialog
    from gc import collect

    # Use file open dialog to obtain file path
    root = Tk()
    opt = dict(defaultextension='.ma',filetypes=[('ACQ4 (HDF5) file','*.ma'), ('All files','*.*')])
    if 'loadcwd' not in globals():
        global loadcwd
    else:
        opt['initialdir'] = loadcwd
    filepath = tkFileDialog.askopenfilename(**opt)
    root.withdraw()
    
    if filepath != '':
        
        # Load data into python
        loadcwd = filepath.rsplit('/',1)[0]
        import ephysIO
        data = ephysIO.MAload(filepath, channel)
        print filepath

        # Display data in Stimfit 
        import stf
        if data.get('yunit') == 'A': 
            stf.new_window_list(1.0e+12 * data.get('array')[1::])
            stf.set_yunits('p'+data.get('yunit'))
        elif data.get('yunit') == 'V': 
            stf.new_window_list(1.0e+3 * data.get('array')[1::])
            stf.set_yunits('m'+data.get('yunit'))
        stf.set_sampling_interval(1.0e+3 * data['xdiff'])
        stf.set_xunits('m'+data.get('xunit'))
        stf.set_trace(0)

        # Import metadata into stimfit
        stf.set_recording_comment('\n'.join(data['notes']))
        date = data['saved'][0:8]
        date = tuple(map(int,(date[0:4],date[4:6],date[6:8])))
        stf.set_recording_date('%s-%s-%s'%date)
        time = data['saved'][9::]
        time = tuple(map(int,(time[0:2],time[2:4],time[4:6])))
        stf.set_recording_time('%i-%i-%i'%time)
        
    else:
        
        data = {}
        
    collect()
    
    return

def loadflex():
    """
    Load raw traces of FlexStation data from CSV files 
    """

    # Import required modules
    from os import chdir
    import csv
    import stf
    import numpy as np
    from Tkinter import Tk
    import tkFileDialog
    from gc import collect

    # Use file open dialog to obtain file path
    root = Tk()
    opt = dict(defaultextension='.csv',filetypes=[('Comma Separated Values file','*.csv'), ('All files','*.*')])
    if 'loadcwd' not in globals():
        global loadcwd
    else:
        opt['initialdir'] = loadcwd
    filepath = tkFileDialog.askopenfilename(**opt)
    root.withdraw()

    if filepath != '':

        # Move to file directory and check file version
        loadcwd = filepath.rsplit('/',1)[0]
        print filepath
        chdir(loadcwd)
        
        # Load data into numpy array 
        with open(filepath,'rb') as csvfile:
            csvtext = csv.reader(csvfile)
            data = []
            for row in csvtext:
                data.append(row)
        data=np.array(data)
        time=data.T[0][1::].astype(np.float)
        sampling_interval = np.mean(np.diff(time))
        comment = 'Temperature: %d degrees Centigrade' % np.mean(data.T[1][1::].astype(np.float))
        
        # Plot fluorescence measurements
        well = data.T[2::,0]
        data=data.T[2::,1::]
        ridx=[]
        idx=[]
        for i in range(96):
            if np.all(data[i]==''):
                ridx.append(i)
            else:
                idx.append(i)
        data = np.delete(data,ridx,0)
        data[[data=='']]='NaN'
        data[[data==' ']]='NaN'
        delrow=np.any(data=='NaN',0)
        didx=[]
        for i in range(np.size(data,1)):
            if np.any(data[::,i]=='NaN',0):
                didx.append(i)
        time=np.delete(time,didx,0)
        data=np.delete(data,didx,1)
        data=data.astype(np.float)
        stf.new_window_list(data)

        # Set x-units and sampling interval 
        stf.set_xunits('ms')
        stf.set_yunits(' ')
        stf.set_sampling_interval(1000*sampling_interval)

        # Record temperature
        comment += '\nTr\tWell'
        for i in range(len(idx)):
            comment += '\n%i\t%s' % (i+1,well[idx[i]])
        comment += '\nInitial time point: %.3g' % time[0]
        print comment
        stf.set_recording_comment(comment)
        
    else:
        
        data = {}
        
    collect()
        
    return

def reverse():
    """
    Reverse the order of all traces
    """

    reversed_traces = []
    n = stf.get_size_channel()
    for i in range(n):
        reversed_traces.append(stf.get_trace(n-1-i))
    stf.new_window_list(reversed_traces)

    return

def blankstim():
    """
    Blank values between fit cursors in all traces in the active channel.
    Typically used to blank stimulus artifacts.
    """

    fit_start = stf.get_fit_start()
    fit_end = stf.get_fit_end()
    blanked_traces = []
    for i in range(stf.get_size_channel()):
        tmp = stf.get_trace(i)
        tmp[fit_start:fit_end] = np.nan
        blanked_traces.append(tmp)
    stf.new_window_list(blanked_traces)

    return


def crop():
    
    si = stf.get_sampling_interval()
    start = stf.get_fit_start()*si
    end = stf.get_fit_end()*si
    spells.cut_sweeps(start,end-start)
    
    return

   
def peakscale():
    """
    Scale the selected traces in the currently active channel to their mean peak amplitude. 

    """

    # Measure baseline in selected traces
    base=[]
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        base.append(stf.get_base())

    # Subtract baseline from selected traces
    stf.subtract_base()

    # Measure peak amplitudes in baseline-subtracted traces
    stf.select_all()
    peak = []
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        peak.append(stf.get_peak())

    # Calculate scale factor to make peak equal to the mean peak amplitude
    scale_factor = peak / np.mean(peak)

    # Scale the traces and apply offset equal to the mean baseline
    scaled_traces = [stf.get_trace(i) / scale_factor[i] + np.mean(base) for i in stf.get_selected_indices()]

    # Close window of baseline-subtracted traces
    stf.close_this()

    return stf.new_window_list(scaled_traces)


def subtract_trace():
    """
    Subtract the selected trace from all traces in the currently active channel

    """

    # Find index of the selected trace to subtract from all the other traces
    idx = stf.get_selected_indices()
    if len(idx)>1:
        raise ValueError('More than one trace was selected')
    elif len(idx)<1:
        raise ValueError('Select one trace to subtract from the others')

    # Apply subtraction
    subtracted_traces = [stf.get_trace(i) - stf.get_trace(idx[0]) for i in range(stf.get_size_channel())]

    return stf.new_window_list(subtracted_traces)


def median_filter(n):
    """
    Perform median smoothing filter on the selected traces. 
    Computationally this is achieved by a central simple moving 
    median over a sliding window of n points.

    The function uses reflect (or bounce) end corrections

    """

    # Check that at least one trace was selected
    if not stf.get_selected_indices():
        raise IndexError('No traces were selected')
    
    # Check that the number of points in the sliding window is odd
    n = int(n)
    if n % 2 != 1:
        raise ValueError('The filter rank must be an odd integer')
    elif n <= 1:
        raise ValueError('The filter rank must > 1')

    # Apply smoothing filter
    filtered_traces = [];
    for i in stf.get_selected_indices():
        l = stf.get_size_trace(i)
        padded_trace = np.pad(stf.get_trace(i),(n-1)/2,'reflect')
        filtered_traces.append([np.median(padded_trace[j:n+j]) for j in range(l)])

    print "Window width was %g ms" % (stf.get_sampling_interval()*(n-1))

    return stf.new_window_list(filtered_traces)


def normalize():
    """
    Normalize to the peak amplitude of the selected trace and 
    scale all other traces in the currently active channel by 
    the same factor. 

    Ensure that you subtract the baseline before normalizing
    """

    # Find index of the selected trace
    idx = stf.get_selected_indices()
    if len(idx)>1:
        raise ValueError('More than one trace was selected')
    elif len(idx)<1:
        raise ValueError('Select one trace to subtract from the others')

    # Measure peak amplitude in the selected trace
    stf.set_trace(idx[0])
    refval = np.abs(stf.get_peak())

    # Apply normalization
    scaled_traces = [stf.get_trace(i) / refval for i in range(stf.get_size_channel())]

    return stf.new_window_list(scaled_traces)


def peakalign():
    """
    Shift the selected traces in the currently active channel to align the peaks. 

    """

    # Measure peak indices in the selected traces
    pidx = []
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        pidx.append(stf.peak_index())

    # Find the earliest peak
    pref = min(pidx)

    # Align the traces
    j = 0
    shifted_traces = []
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        shift = int(pref-pidx[j])
        shifted_traces.append(np.roll(stf.get_trace(),shift))                     
        j += 1

    return stf.new_window_list(shifted_traces)


def risealign():
    """
    Shift the selected traces in the currently active channel to align to the rise. 

    """

    # Measure peak indices in the selected traces
    rtidx = []
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        rtidx.append(stf.rtlow_index())

    # Find the earliest peak
    rtref = min(rtidx)

    # Align the traces
    j = 0
    shifted_traces = []
    for i in stf.get_selected_indices():
        stf.set_trace(i)
        shift = int(round(rtref-rtidx[j]))
        shifted_traces.append(np.roll(stf.get_trace(),shift))                     
        j += 1

    return stf.new_window_list(shifted_traces)


def chebexp(n):
    """
    Fits sums of exponentials with offset to the current trace in the
    active channel using the Chebyshev tranform algorithm.

    Reference:
    Malachowski, Clegg and Redford (2007) J Microsc 228(3): 282-95
    """
    
    # Get data trace between fit/decay cursors
    y = np.double(stf.get_trace()[stf.get_fit_start():stf.get_fit_end()])
    si = stf.get_sampling_interval()
    l = len(y)
    N = np.double(l-1)

    # Calculate time dimension with unit 1
    t = np.arange(0,l,1,np.double)

    # Calculate the maximum order Chebyshev polynomials to generate
    if l > 30:
        Tn = 30
    else:
        Tn = N

    # Generate the polynomials T and coefficients d
    T0 = np.ones((l),np.double)
    R0 = np.sum(T0**2) 
    d0 = np.sum((T0*y)/R0)
    T = np.zeros((l,Tn),np.double)
    T[:,0] = 1-2*t/N
    T[:,1] = 1-6*t/(N-1)+6*t**2/(N*(N-1))
    R = np.zeros((Tn),np.double)
    d = np.zeros((Tn),np.double)
    for j in range(Tn):
        if j > 1:
            A = (j+1)*(N-j) 
            B = 2*(j+1)-1 
            C = j*(N+j+1)
            T[:,j] = (B*(N-2*t)*T[:,j-1]-C*T[:,j-2])/A
        R[j] = np.sum(T[:,j]**2)
        d[j] = np.sum(T[:,j]*y/R[j])

    # Generate additional coefficients dn that describe the relationship
    # between the Chebyshev coefficients d and the constant k, which is
    # directly related to the exponent time constant
    dn = np.zeros((n,Tn),np.double)
    for i in range(1,n+1):
        for j in range(1+i,Tn-i+1):
            if i > 1:
                dn[i-1,j-1] = (((N+j+2)*dn[i-2,j]/(2*j+3))-dn[i-2,j-1]-((N-j+1)*dn[i-2,j-2]/(2*j-1)))/2
            else:
                dn[i-1,j-1] = (((N+j+2)*d[j]/(2*j+3))-d[j-1]-((N-j+1)*d[j-2]/(2*j-1)))/2
    for i in range(n):
        dn[i,:] = dn[i,:]*np.double(np.all(dn,0))

    # Form the regression model to find the time constants of each exponent
    Mn = np.zeros((n,n),np.double)
    b = np.zeros(n,np.double)
    for i in range(n):
        b[i] = np.sum(d*dn[i,:])
        for m in range(n):
            Mn[i,m] = -np.sum(dn[i,:]*dn[m,:])

    # Solve the linear problem
    try:
        x = np.linalg.solve(Mn,b)
    except:
        x = np.linalg.lstsq(Mn,b)[0]
    k = np.roots(np.hstack((1,x)))
    if any(k!=np.real(k)):
        raise ValueError("Result is not a sum of %d real exponents" % n) 
    tau = -1/np.log(1+k)

    # Generate the Chebyshev coefficients df for each exponent
    df0 = np.zeros(n,np.double)
    df = np.zeros((n,Tn),np.double)
    for i in range(n):
        for j in range(Tn):
            df[i,j] = np.sum(np.exp(-t/tau[i])*T[:,j]/R[j])
        df0[i] = np.sum(np.exp(-t/tau[i])*T0/R0)

    # Form the regression model to find the amplitude of each exponent
    Mf = np.zeros((n,n),np.double)
    b = np.zeros(n,np.double)
    for i in range(n):
        b[i] = np.sum(d*df[i,:])
        for m in range(n):
            Mf[i,m] = np.sum(df[i,:]*df[m,:])

    # Solve the linear problem
    try:
        a = np.linalg.solve(Mf,b)
    except:
        a = np.linalg.lstsq(Mf,b)[0]

    # Calculate the offset for the fit
    offset = d0-np.sum(df0*a.T)

    # Prepare output
    retval = [("Amp_%d"%i,a[i]) for i in range(n)]
    retval += [("Tau_%d"%i,si*tau[i]) for i in range(n)]
    retval += [("Offset",offset)]
    retval = dict(retval)

    return retval


def monoexpfit():
    """
    Fits monoexponential function with offset to data between the fit cursors
    in the current trace of the active channel using the Levenberg-Marquardt
    algorithm. Requires Scipy.
    """
    
    # Get data
    fit_start = stf.get_fit_start()
    fit_end = stf.get_fit_end()
    y = np.double(stf.get_trace()[fit_start:fit_end])
    si = stf.get_sampling_interval()
    l = len(y)
    t = si*np.arange(0,l,1,np.double)

    # Define monoexponential function
    def f(t,*p): return p[0]+p[1]*np.exp(-t/p[2])

    # Get initial values from Chebyshev transform fit
    init = chebexp(1)
    p0  = (init.get('Offset'),)
    p0 += (init.get('Amp_0'),)
    p0 += (init.get('Tau_0'),)
  
    # Optimize fit using Levenberg-Marquardt algorithm
    options = {"ftol":2.22e-16,"xtol":2.22e-16,"gtol":2.22e-16}
    [p, pcov] = optimize.curve_fit(f,t,y,p0,**options)
    fit = f(t,*p)

    # Calculate SSE
    SSE = np.sum((y-fit)**2)

    # Plot fit in a new window
    matrix = np.zeros((2,stf.get_size_trace()))*np.nan
    matrix[0,:] = stf.get_trace()
    matrix[1,fit_start:fit_end] = fit
    stf.new_window_matrix(matrix)
    
    # Create table of results 
    retval  = [("p0_Offset",p[0])]
    retval += [("p1_Amp_0",p[1])]
    retval += [("p2_Tau_0",p[2])]
    retval += [("SSE",SSE)]
    retval += [("dSSE",1.0-np.sum((y-f(t,*p0))**2)/SSE)]
    retval += [("Time fit begins",fit_start*si)]
    retval += [("Time fit ends",fit_end*si)]
    retval = dict(retval)
    stf.show_table(retval,"monoexpfit, Section #%i" % float(stf.get_trace_index()+1))
    
    return


def biexpfit():
    """
    Fits biexponential function with offset to data between the fit cursors
    in the current trace of the active channel using the Levenberg-Marquardt
    algorithm. Requires Scipy.
    """
    
    # Get data
    fit_start = stf.get_fit_start()
    fit_end = stf.get_fit_end()
    y = np.double(stf.get_trace()[fit_start:fit_end])
    si = stf.get_sampling_interval()
    l = len(y)
    t = si*np.arange(0,l,1,np.double)

    # Define biexponential function
    def f(t,*p): return p[0]+p[1]*np.exp(-t/p[2])+p[3]*np.exp(-t/p[4])

    # Get initial values from Chebyshev transform fit
    init = chebexp(2)
    p0  = (init.get('Offset'),)
    p0 += (init.get('Amp_0'),)
    p0 += (init.get('Tau_0'),)
    p0 += (init.get('Amp_1'),)
    p0 += (init.get('Tau_1'),)
  
    # Optimize fit using Levenberg-Marquardt algorithm
    options = {"ftol":2.22e-16,"xtol":2.22e-16,"gtol":2.22e-16}
    [p, pcov] = optimize.curve_fit(f,t,y,p0,**options)
    fast = p[0]+p[1]*np.exp(-t/p[2])
    slow = p[0]+p[3]*np.exp(-t/p[4])
    wfit = f(t,*p)

    # Calculate SSE
    SSE = np.sum((y-wfit)**2)

    # Calculate weighted time constant
    wtau = p[1]/(p[1]+p[3])*p[2] + p[3]/(p[1]+p[3])*p[4]

    # Plot fit and both components in a new window
    matrix = np.zeros((4,stf.get_size_trace()))*np.nan
    matrix[0,:] = stf.get_trace()
    matrix[1,fit_start:fit_end] = wfit
    matrix[2,fit_start:fit_end] = fast
    matrix[3,fit_start:fit_end] = slow
    stf.new_window_matrix(matrix)
    
    # Create table of results 
    retval  = [("p0_Offset",p[0])]
    retval += [("p1_Amp_0",p[1])]
    retval += [("p2_Tau_0",p[2])]
    retval += [("p3_Amp_1",p[3])]
    retval += [("p4_Tau_1",p[4])]
    retval += [("SSE",SSE)]
    retval += [("dSSE",1.0-np.sum((y-f(t,*p0))**2)/SSE)]
    retval += [("Weighted tau",wtau)]
    retval += [("Time fit begins",fit_start*si)]
    retval += [("Time fit ends",fit_end*si)]
    retval = dict(retval)
    stf.show_table(retval,"biexpfit, Section #%i" % float(stf.get_trace_index()+1))
    
    return


def raster(event_times_list, color='k'):
    """
    Creates a raster plot

    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines

    Returns
    -------
    ax : an axis containing the raster plot

    Example usage
    -------
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = penn.raster(spikes)
    plt.title('Raster plot')
    plt.xlabel('Time')
    plt.ylabel('Trial')
    fig.show()
    """
    import matplotlib.pyplot as plt
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax


def rmeantraces(binwidth):
    """
    Perform running mean of all traces in the active channel. 
    The number of traces averaged is defined by binwidth. 
    """

    n = binwidth
    N = stf.get_size_channel()
    m = N/n
    if np.fix(m)!=m:
        raise ValueError('The number of traces is not divisible by n')

    # loop index calculations: [[i*n+j for j in range(n)] for i in range(m)]
    binned_traces = [[stf.get_trace(i*n+j) for j in range(n)] for i in range(m)]
    mean_traces = [np.mean(binned_traces[i],0) for i in range(m)]

    return stf.new_window_list(mean_traces)


def rmean3traces():
    """
    Perform running mean of all traces in the active channel. 
    The number of traces averaged is 3. 
    """

    return rmeantraces(3)

def yoffset(value):
    """
    Apply a common offset to all traces in the currently active channel.
    """

    offset_traces = [stf.get_trace(i) + value for i in range(stf.get_size_channel())]

    return stf.new_window_list(offset_traces)

def trainpeaks():
    """    
    Measure a 20 Hz train of peaks starting at 260 ms into the trace 
    """

    pk = []
    for i in range(5):       
        stf.set_base_start(int(255/stf.get_sampling_interval())+(50/stf.get_sampling_interval())*i)
        stf.set_base_end(int(259/stf.get_sampling_interval())+(50/stf.get_sampling_interval())*i)
        stf.set_peak_start(int(260.5/stf.get_sampling_interval())+(50/stf.get_sampling_interval())*i)
        stf.set_peak_end(int(270.5/stf.get_sampling_interval())+(50/stf.get_sampling_interval())*i)
        stf.measure()
        pk.append(stf.get_peak()-stf.get_base())

    # Create table of results 
    dictlist  = [("Peak 1",pk[0])]
    dictlist += [("Peak 2",pk[1])]
    dictlist += [("Peak 3",pk[2])]
    dictlist += [("Peak 4",pk[3])]
    dictlist += [("Peak 5",pk[4])]    
    retval = dict(dictlist)
    stf.show_table(retval,"peaks, Section #%i" % float(stf.get_trace_index()+1))

    # Create table of results 
    dictlist  = [("Peak 1",pk[0]/pk[0]*100)]
    dictlist += [("Peak 2",pk[1]/pk[0]*100)]
    dictlist += [("Peak 3",pk[2]/pk[0]*100)]
    dictlist += [("Peak 4",pk[3]/pk[0]*100)]
    dictlist += [("Peak 5",pk[4]/pk[0]*100)]    
    retval = dict(dictlist)
    stf.show_table(retval,"norm peaks, Section #%i" % float(stf.get_trace_index()+1))

    return

def mean_every_Nth(N):
    """
    Perform mean of the first and every Nth trace
    """

    m = stf.get_size_channel()/(N-1)
    if np.fix(m)!=m:
        raise ValueError('The number of traces is not divisible by N')

    # loop index calculations: [[i*n+j for j in range(n)] for i in range(m)]
    binned_traces = [[stf.get_trace((i+1)+j*(N-1)-1) for j in range(m)] for i in range(N-1)]
    mean_traces = [np.mean(binned_traces[i],0) for i in range(N-1)]

    return stf.new_window_list(mean_traces)

def mean_every_9th():
    """
    Perform mean of the first and every 9th trace
    """

    return mean_every_Nth(9)

def SBR():
    """
    Calculate signal-to-baseline ratio (SBR) or delta F / F0 for
    traces in the active window. The result is expressed as a %.

    Ensure that the baseline cursors are positioned appropriately.
    """

    SBR_traces = [100*(stf.get_trace(i)-stf.get_base())/stf.get_base() for i in range(stf.get_size_channel())]
    stf.new_window_list(SBR_traces)
    stf.set_yunits('%')
    
    return 

def multiscale_traces(multiplier_list):
    """
    Scale each trace to the respective multiplier in the list argument
    """

    if len(multiplier_list)!=stf.get_size_channel():
        raise ValueError('The number of multipliers and traces are not equal')
    scaled_traces = [stf.get_trace(i)*multiplier_list[i] for i in range(stf.get_size_channel())]
    
    return stf.new_window_list(scaled_traces)

def upsample_flex():
    """
    Upsample to sampling interval of 1 ms using cubic spline interpolation
    """
    
    old_time = [i*stf.get_sampling_interval() for i in range(stf.get_size_trace())]
    new_time = range(int(np.fix((stf.get_size_trace()-1)*stf.get_sampling_interval())))
    new_traces = []
    for i in range(stf.get_size_channel()):
        f=interpolate.interp1d(old_time,stf.get_trace(i),'cubic')
        new_traces.append(f(new_time))
    stf.new_window_list(new_traces)
    stf.set_sampling_interval(1)
    
    return 

def batch_integration():
    """
    Perform batch integration between the decay/fit cursors of all traces
    in the active window
    """
    n = int(stf.get_fit_end()+1-stf.get_fit_start())
    x = [i*stf.get_sampling_interval() for i in range(n)]
    dictlist = []
    for i in range(stf.get_size_channel()):
        stf.set_trace(i)
        y = stf.get_trace()[int(stf.get_fit_start()):int(stf.get_fit_end()+1)]
        auc = np.trapz(y-stf.get_base(),x)
        dictlist += [("%i" % (i+1), auc)] 
    retval = dict(dictlist)
    stf.show_table(retval,"Area Under Curve")
    stf.set_trace(0)

    return 
