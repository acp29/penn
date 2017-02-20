# Penn lab python module for ACQ4
# Version 11 May 2015

def SCE_200Hz():
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
    for i in range(5):
        os.system("pause") 
        tr.runSingle(store=False)
    print("Completed!")
    b=time.time()   # Finish time
    print("Time elapsed: %g seconds") % (b-a)
    winsound.Beep(800,1000) 
    del(a,b)

def SCE_100Hz():
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
    for i in range(5):
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

def test():
    """
    Single-cell electroporation
    """
    import time
    import os 
    from acq4.Manager import getManager
    from acq4.util.DataManager import getHandle
    man = getManager()
    tr = man.getModule('Task Runner')
    protocol = getHandle('C:\\Users\\Public\\Documents\\acq4 Settings\\config\\protocols\\acp29\\SCE\\SCE_100Hz')
    tr.loadTask(protocol)
    tr.runSingle(store=False)
    from acq4.Manager import getManager
    from acq4.util.DataManager import getHandle
    man = getManager()
    tr = man.getModule('Task Runner')
    protocol = getHandle('C:\\Users\\Public\\Documents\\acq4 Settings\\config\\protocols\\acp29\\SCE\\SCE_200Hz')
    tr.loadTask(protocol)
    tr.runSingle(store=False)
