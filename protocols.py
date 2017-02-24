## Penn lab python module for ACQ4
## Version 23 February 2017

def SCE_200Hz(n=5):
    """
    Single-cell electroporation
    """
    import time
    import os 
    import winsound
    from acq4.Manager import getManager
    from acq4.util.DataManager import getHandle
    print("Make terminal window the active window")
    a=time.time()   # Start time
    man = getManager()
    tr = man.getModule('Task Runner')
    protocol = getHandle('C:\\Users\\Public\\Documents\\acq4 Settings\\config\\protocols\\acp29\\SCE\\SCE_200Hz')
    tr.loadTask(protocol)
    for i in range(n):
        os.system("pause") 
        tr.runSingle(store=False)
    print("Completed!")
    b=time.time()   # Finish time
    print("Time elapsed: %g seconds") % (b-a)
    winsound.Beep(800,1000) 
    del(a,b)

def SCE_100Hz(n=5):
    """
    Single-cell electroporation
    """
    import time
    import os 
    import winsound
    from acq4.Manager import getManager
    from acq4.util.DataManager import getHandle
    print("Make terminal window the active window")
    a=time.time()   # Start time
    man = getManager()
    tr = man.getModule('Task Runner')
    protocol = getHandle('C:\\Users\\Public\\Documents\\acq4 Settings\\config\\protocols\\acp29\\SCE\\SCE_100Hz')
    tr.loadTask(protocol)
    for i in range(n):
        os.system("pause") 
        tr.runSingle(store=False)
    print("Completed!")
    b=time.time()   # Finish time
    print("Time elapsed: %g seconds") % (b-a)
    winsound.Beep(800,1000) 
    del(a,b)

def raiseVm():
    import time
    import winsound
    from acq4.Manager import getManager
    man = getManager()
    clamp1 = man.getDevice('Clamp1')
    clamp2 = man.getDevice('Clamp2')
    clamp1.setMode('VC')
    clamp2.setMode('VC')
    clamp1.setHolding('VC',-0.065)
    clamp2.setHolding('VC',-0.065)
    for i in range(95):
        delta = (i+1)*0.001    # Raise Vm at a rate of 1 mV/s
        clamp1.setHolding('VC',-0.065+delta)
        clamp2.setHolding('VC',-0.065+delta)
        time.sleep(1)
    clamp1.setHolding('VC',0.030)
    clamp2.setHolding('VC',0.030)
    winsound.Beep(800,1000)

def NAM(store=True,interval=30,lag=3,initialize=False):
    # Import required python modules
    import os
    import time
    import thread
    from acq4.Manager import getManager
    from acq4.util.DataManager import getHandle

    # Get valve devices
    man=getManager()
    V1=man.getDevice('V1')
    V2=man.getDevice('V2')
    V3=man.getDevice('V3')
    V4=man.getDevice('V4')
    V5=man.getDevice('V5')
    V6=man.getDevice('V6')
    V7=man.getDevice('V7')
    V8=man.getDevice('V8')

    # Prepare task for acquisition
    tr=man.getModule('Task Runner')
    protocol=getHandle('C:\\Users\\Public\\Documents\\acq4 Settings\\config\\protocols\\wa62\\500ms')
    tr.loadTask(protocol)
    tr.protoStateGroup.setState({'cycleTime':interval,'repetitions':6})

    # Define a function for the acquisition
    def glu500(tr):
        tr.runSequence(store)
        
    # Define a function for the solution switches
    def switch(tr,V1,V2,V3,V4,V5,V6,interval):
        V8.setChanHolding('8',1)
        for i in range(6):
            if flag!=0:
                eval('V'+str(i+1)).setChanHolding(str(i+1),1)
                for j in range(interval):
                    time.sleep(1)
                    if flag==0:
                        break
                eval('V'+str(i+1)).setChanHolding(str(i+1),0)
        V8.setChanHolding('8',0)
        tr.stopSequence()

    # Define a function to monitor keyboard input
    def monitor_keyboard():
        global flag
        flag=1
        flag=os.system("pause")

    # Define a function to coordinate the acquisition and solution switches
    def run_protocol(tr,V1,V2,V3,V4,V5,V6,interval,lag):
        thread.start_new_thread(monitor_keyboard,())
        thread.start_new_thread(switch,(tr,V1,V2,V3,V4,V5,V6,interval,))
        for k in range(interval-lag):
            time.sleep(1)
            if flag==0:
                break
        if flag!=0:
            glu500(tr)

    # Initialise valves
    if initialize is True:
        for i in range(8):
            eval('V'+str(i+1)).setChanHolding(str(i+1),0)
            
    # Run the protocol
    # To terminate at any time, make terminal window the active window and press any key
    run_protocol(tr,V1,V2,V3,V4,V5,V6,interval,lag)
